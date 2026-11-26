"""Distill the offline macro teacher using only the inputs available during play."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .curriculum_actor import Residual
from .deck_training import funnel
from .heart_curriculum import read_model, save_model
from .recovery import panel
from .train import write


def comparisons(files):
    pairs = []
    for path in files:
        with np.load(path) as data:
            rows = data["examples"]
        starts = np.r_[0, np.flatnonzero(np.any(rows[1:,:2] != rows[:-1,:2], axis=1))+1]
        parent = np.repeat(starts, np.diff(np.r_[starts, len(rows)]))
        index = np.flatnonzero(parent != np.arange(len(rows)))
        # The same seed and state are used for both continuations. Their difference is the label.
        outcome = rows[:,6] + 5*rows[:,5] + 15*rows[:,4] + .2*rows[:,7]
        change = np.clip((outcome[index]-outcome[parent[index]])/7., -4., 4.)
        pairs.append((rows[index,3], rows[index,9:], rows[parent[index],3], rows[parent[index],9:], change))
    return tuple(torch.from_numpy(np.concatenate(col)) for col in zip(*pairs))


def residual(model, item, feature):
    x = model.layer(feature) + model.item(item.long()) + model.foe.weight[0]
    return 8*torch.tanh(model.out(x.relu()).flatten()/8)


def predict(model, data, index):
    item, feature, parent, context, _ = data
    return residual(model,item[index],feature[index]) - residual(model,parent[index],context[index])


def metrics(model, data):
    with torch.no_grad():
        predicted = predict(model,data,slice(None)).numpy()
    target = data[-1].numpy()
    moved = target != 0
    rescue=target>0
    proposed=predicted>0
    return dict(pairs=len(target),non_ties=int(moved.sum()),
                mse=float(np.mean((predicted-target)**2)),zero_mse=float(np.mean(target**2)),
                non_tie_sign_accuracy=float(np.mean((predicted[moved]>0)==(target[moved]>0))),
                always_prefer_parent_accuracy=float(np.mean(target[moved]<0)),
                beneficial_alternatives=int(rescue.sum()),
                positive_recall=float(np.mean(proposed[rescue])),
                positive_precision=float((proposed&rescue).sum()/max(1,proposed.sum())),
                correlation=float(np.corrcoef(predicted,target)[0,1]))


def fit(training,valid,rescue_weight):
    torch.manual_seed(17403)
    model=Residual()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.014)
    history=[]
    best=float("inf")
    for epoch in range(1,33):
        for ids in torch.randperm(len(training[-1])).split(1024):
            predicted=predict(model,training,ids)
            y=training[-1][ids]
            losses=F.smooth_l1_loss(predicted,y,beta=1.,reduction="none")
            loss=(losses*torch.where(y>0,rescue_weight,1.)).mean()
            optimizer.zero_grad();loss.backward();optimizer.step()
        if epoch in (1,4,8,16,32):
            check=metrics(model,valid)
            history.append(dict(epoch=epoch,**check))
            if check["mse"]<best:
                best=check["mse"]
                config=model.export()
                chosen_epoch=epoch
    return config,history,chosen_epoch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train",type=int,default=173000000)
    p.add_argument("--runs",type=int,default=1024)
    p.add_argument("--valid",type=int,default=172000000)
    p.add_argument("--dir",type=Path,default=Path("build/macro-teacher"))
    p.add_argument("--out",type=Path,default=Path("build/macro-student"))
    p.add_argument("--threads",type=int,default=10)
    p.add_argument("--screen-only",action="store_true")
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    training=comparisons([args.dir/f"{args.train+i}.npz" for i in range(0,args.runs,32)])
    valid=comparisons([args.dir/f"{args.valid+i}.npz" for i in range(0,256,32)])
    torch.set_num_threads(3)
    parent=read_model("build/trajectory-teacher/challenger.json.gz")
    report=dict(training_pairs=len(training[-1]),fits=[],validation=[],test={})
    finalists=[]
    for weight in (1.,6.):
        config,history,epoch=fit(training,valid,weight)
        report["fits"].append(dict(rescue_weight=weight,history=history,epoch=epoch))
        for scale,scope in ((.5,5),(2,5),(5,5),(2,3),(5,3),(12,3)):
            model={**parent,"curriculum_actor":[*config[:5],scale,scope]}
            stats,rows=panel(model,174000000,4096,args.threads)
            result={**stats,**funnel(rows),"scale":scale,"scope":scope,"weight":weight,"epoch":epoch}
            report["validation"].append(result)
            finalists.append((model,result))
    # Sparse Heart encounters are too noisy on the tuning panel. Use late-boss arrivals
    # plus Act 3 arrivals as a secondary survival signal, fixed before evaluation.
    def progress(pair):
        s=pair[1]
        return s["first_boss"] + 2*s["second_boss"] + .1*s["act3"] + 4*s["act4"]
    candidate,chosen=max(finalists,key=progress)
    report["scale"]=chosen["scale"]
    report["scope"]=chosen["scope"]
    save_model(args.out/"challenger.json.gz",candidate)
    report["selected"]=chosen
    print("selected",chosen,flush=True)
    if args.screen_only:
        write(args.out/"report.json",report)
        return
    for name,policy in (("parent",parent),("student",candidate)):
        stats,rows=panel(policy,175000000,32768,args.threads)
        report["test"][name]={**stats,**funnel(rows)}
        print(name, report["test"][name],flush=True)
    write(args.out/"report.json",report)


if __name__=="__main__":
    main()
