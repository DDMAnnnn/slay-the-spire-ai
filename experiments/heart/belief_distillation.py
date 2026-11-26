"""Distill a public-information combat search and reuse its student inside the search."""
import argparse
import json
from pathlib import Path

import heart_native as engine
import numpy as np
import torch
from torch.nn import functional as F

from .choice_attention import ActionSet,Groups
from .heart_curriculum import read_model,save_model
from .inventory_search import panel_chunks
from .public_belief import evaluate
from .replay import apply_model
from .train import write


def collect(out,w,start,n,threads):
    paths=[];totals=np.zeros(3,int)
    for offset in range(0,n,32):
        path=out/f"{start+offset}.npz"
        paths.append(path)
        if not path.exists():
            rows,counts=engine.belief_examples(w,start+offset,min(32,n-offset),threads)
            tmp=path.with_suffix(".tmp.npz");np.savez_compressed(tmp,rows=rows,counts=counts);tmp.replace(path)
        with np.load(path) as z:totals+=z["counts"]
    return paths,dict(zip(("matched_states","changed_actions","sampled_states"),map(int,totals)))


def outputs(model,rows,mask):
    raw=model(rows,mask)[:,:,0]
    # This expression is also used by the native attention scorer on a fresh play.
    correction=24*torch.tanh((raw-raw[:,:1])/8.)
    targets=rows[:,:,2]
    gap=rows[:,:,4]
    return (gap+correction).masked_fill(~mask,-1e8),(gap+targets).masked_fill(~mask,-1e8),correction


def validate(model,groups,device):
    totals=np.zeros(5)
    with torch.no_grad():
        for ix in torch.arange(len(groups.starts),device=device).split(384):
            rows,mask,_=groups.batch(ix)
            scores,target,_=outputs(model,rows,mask)
            i=scores.argmax(1,keepdim=True)
            best=target.max(1).values
            regret=best-target.gather(1,i).flatten()
            baseline=best-target[:,0]
            changed=baseline>1e-5
            totals+=[len(ix),float(regret.clamp(max=24).sum()),float(baseline.clamp(max=24).sum()),
                     int(changed.sum()),int((changed & (regret<1e-5)).sum())]
    return dict(groups=int(totals[0]),teacher_regret=float(totals[1]/totals[0]),
                incumbent_regret=float(totals[2]/totals[0]),changed=int(totals[3]),
                correct_changes=int(totals[4]))


def fit(train,valid,device):
    torch.manual_seed(19607);torch.set_num_threads(2)
    net=ActionSet(True).to(device)
    optimizer=torch.optim.AdamW(net.parameters(),lr=.0013,weight_decay=.01)
    best=float("inf");history=[]
    for epoch in range(1,41):
        for ix in torch.randperm(len(train.starts),device=device).split(384):
            rows,mask,_=train.batch(ix)
            pred,target,delta=outputs(net,rows,mask)
            target_prob=(target/2).softmax(1)
            kl=F.kl_div((pred/2).log_softmax(1),target_prob,reduction="none").sum(1)
            mse=(F.smooth_l1_loss(delta,rows[:,:,2].clamp(-24,24),reduction="none")*mask).sum(1)/mask.sum(1)
            changed=target.max(1).values>target[:,0]+1e-5
            loss=((kl+.07*mse)*(1.+2*changed)).mean()
            optimizer.zero_grad();loss.backward();optimizer.step()
        if epoch in (1,4,10,20,40):
            stats=validate(net,valid,device)
            history.append(dict(epoch=epoch,**stats))
            if stats["teacher_regret"]<best:
                best=stats["teacher_regret"];chosen=epoch
                config=net.export([1.,0.,0.,0.])
    return config,dict(epoch=chosen,history=history)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=Path("build/belief-student"))
    p.add_argument("--threads",type=int,default=4)
    p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    parent=read_model("build/trajectory-teacher/challenger.json.gz")
    cfg=json.loads(Path("build/public-belief/report.json").read_text())["selected"]
    teacher={**parent,"public_belief":cfg}
    w=apply_model("first",teacher)
    train,train_info=collect(args.out,w,196000000,1536,args.threads)
    valid,valid_info=collect(args.out,w,197000000,256,args.threads)
    config,fit_info=fit(Groups(train,args.device),Groups(valid,args.device),args.device)
    report=dict(train=train_info,valid=valid_info,fit=fit_info,selection=[],test={})
    engine.set_belief(0,0,0.)
    choices=[]
    for scale in (.7,1.5,3.,6.):
        m={**parent,"choice_attention":[*config,scale]}
        apply_model("first",m);s=evaluate(w,198000000,2048,args.threads)
        report["selection"].append(dict(scale=scale,**s))
        rank=s["first_boss"]+2*s["second_boss"]+.1*s["act3"]+4*s["act4"]
        choices.append((rank,m))
    _,student=max(choices,key=lambda x:x[0])
    save_model(args.out/"student.json.gz",student)
    combined={**student,"public_belief":cfg}
    save_model(args.out/"search-student.json.gz",combined)
    write(args.out/"fit.json",report)
    # The independent inventory-search experiment is already playing both controls on
    # this exact range. Both students are frozen before we open those comparisons.
    report["panel_start"]=194000000
    report["controls"]="build/paired-inventory-search/report.json (parent and belief)"
    for name,use_search in (("student",False),("search_student",True)):
        engine.set_choice_attention(*student["choice_attention"])
        engine.set_belief(cfg["draws"] if use_search else 0,cfg["scope"],cfg["scale"])
        report["test"][name]=panel_chunks(args.out,name,w,194000000,32768,args.threads)
        write(args.out/"report.json",report)
        print(name,report["test"][name],flush=True)


if __name__=="__main__":
    main()
