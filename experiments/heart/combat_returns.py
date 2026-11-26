"""Learn public action scores from the survival, HP, enemy HP and potion result of each move."""
import argparse
from pathlib import Path

import heart_native as engine
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .curriculum_actor import Residual
from .deck_training import funnel
from .heart_curriculum import read_model, save_model
from .recovery import panel
from .replay import apply_model
from .train import write


def collect(out,weights,start,n,threads,rescue=False):
    arrays=[]
    groups=0
    scouts=[]
    for offset in range(0,n,64):
        seed=start+offset
        path=out/f"{seed}.npz"
        if path.exists():
            with np.load(path) as z:
                raw=z["rows"]
                if rescue: scouts.append(z["stats"])
        else:
            if rescue:
                raw,stats=engine.rescue_combat(weights,seed,min(n-offset,64),threads)
                scouts.append(np.asarray(stats))
                np.savez_compressed(path,rows=raw,stats=stats)
            else:
                raw=engine.combat_returns(weights,seed,min(n-offset,64),threads)
                np.savez_compressed(path,rows=raw)
        starts=np.r_[0,1+np.flatnonzero(np.any(raw[1:,:2]!=raw[:-1,:2],axis=1))]
        parents=np.repeat(starts,np.diff(np.r_[starts,len(raw)]))
        other=np.flatnonzero((np.arange(len(raw))!=parents) & (raw[:,51]>=-7))
        choices=raw[other,3:].copy()
        priors=raw[parents[other],3:].copy()
        arrays.append((choices,priors))
        groups+=len(starts)
    info={}
    if rescue:
        s=np.concatenate(scouts)
        info=dict(scouted=len(s),fatal_fights=int(s[:,0].sum()),rescued=int(s[:,6].sum()),
                  overrides=int(s[:,5].sum()),capped=int(s[:,9].sum()),by_act={})
        for act in (1,2,3,4):
            subset=s[s[:,1]==act]
            info["by_act"][act]=[len(subset),int(subset[:,6].sum())]
    return [torch.from_numpy(np.concatenate(col)) for col in zip(*arrays)],groups,info


def multi_prediction(model,heads,row):
    hidden=model.layer(row[:,8:])+model.item(row[:,0].long())+model.foe(row[:,1].long())
    return heads(hidden.relu())


def labels(rows):
    outcomes=rows[:,2:6].clone()
    outcomes[:,1]/=80.
    return outcomes


def fit(training,validation):
    torch.manual_seed(17741)
    torch.set_num_threads(3)
    model=Residual()
    heads=nn.Linear(20,4)
    opt=torch.optim.AdamW([*model.parameters(),*heads.parameters()],lr=.0014,weight_decay=.01)
    y,prior_y=map(labels,training)
    vy,vprior_y=map(labels,validation)
    history=[]
    best=float("inf")
    for epoch in range(1,33):
        for ids in torch.randperm(len(y)).split(1024):
            predicted=multi_prediction(model,heads,training[0][ids])
            prior=multi_prediction(model,heads,training[1][ids])
            pair_loss=F.smooth_l1_loss(predicted-prior,y[ids]-prior_y[ids],beta=.2)
            absolute_loss=F.smooth_l1_loss(predicted,y[ids],beta=.2)
            loss=pair_loss+.25*absolute_loss
            opt.zero_grad();loss.backward();opt.step()
        if epoch in (1,4,8,16,32):
            with torch.no_grad():
                a,b=(multi_prediction(model,heads,x) for x in validation)
                target=vy-vprior_y
                pred=a-b
                errors=(pred-target).square().mean(0)
                weighted=(pred-target)@torch.tensor([6.,6.,-2.,-.8])
                metric=float(weighted.square().mean())
                history.append(dict(epoch=epoch,paired_mse=errors.tolist(),
                                    zero_change_mse=target.square().mean(0).tolist(),selection_error=metric))
                if metric<best:
                    best=metric
                    selected=epoch
                    params=model.export()
                    readout=heads.weight.detach().clone()
    return params,readout,selected,history


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=Path("build/combat-returns"))
    p.add_argument("--train",type=int,default=2048)
    p.add_argument("--valid",type=int,default=384)
    p.add_argument("--threads",type=int,default=8)
    p.add_argument("--rescue",action="store_true",help="Add practice from the fights that ended natural runs")
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    parent=read_model("build/trajectory-teacher/challenger.json.gz")
    w=apply_model("first",parent)
    ordinary=Path("build/combat-returns") if args.rescue else args.out
    training,tg,_=collect(ordinary,w,176000000,args.train,args.threads)
    validation,vg,_=collect(ordinary,w,177000000,args.valid,args.threads)
    targeted={}
    if args.rescue:
        extra,eg,targeted["train"]=collect(args.out,w,180000000,4096,args.threads,True)
        hold,hg,targeted["validation"]=collect(args.out,w,181000000,640,args.threads,True)
        training=[torch.cat(pair) for pair in zip(training,extra)]
        validation=[torch.cat(pair) for pair in zip(validation,hold)]
        tg+=eg;vg+=hg
        print("rescues",targeted,flush=True)
    print("matched training and validation decisions",tg,vg,flush=True)
    params,readout,epoch,history=fit(training,validation)
    report=dict(training_states=tg,validation_states=vg,training_pairs=len(training[0]),
                validation_pairs=len(validation[0]),chosen_epoch=epoch,history=history,
                targeted=targeted,validation=[],test={})
    options=[]
    for utility in ([6.,6.,-2.,-.8],[10.,3.,-3.,-.4],[4.,9.,-1.,-1.6]):
        # The shared hidden units allow one matrix product to merge the four learned heads.
        # The four intercepts cancel when the native actor subtracts the parent's predicted value.
        projected=(torch.tensor(utility)@readout).tolist()
        for scale in (1.,4.,12.):
            candidate={**parent,"curriculum_actor":[*params[:4],projected,scale,4]}
            s,r=panel(candidate,182000000 if args.rescue else 178000000,3072,args.threads)
            result={**s,**funnel(r),"scale":scale,"utility":utility}
            report["validation"].append(result);options.append((candidate,result))
    def progress(pair):
        s=pair[1]
        return s["first_boss"]+2*s["second_boss"]+.1*s["act3"]+4*s["act4"]
    candidate,chosen=max(options,key=progress)
    report["selected"]=chosen
    save_model(args.out/"challenger.json.gz",candidate)
    models=[("parent",parent),("combat_returns",candidate)]
    if args.rescue: models.append(("ordinary_only",read_model(ordinary/"challenger.json.gz")))
    for name,m in models:
        s,r=panel(m,183000000 if args.rescue else 179000000,32768,args.threads)
        report["test"][name]={**s,**funnel(r)}
        print(name,report["test"][name],flush=True)
    write(args.out/"report.json",report)


if __name__=="__main__":
    main()
