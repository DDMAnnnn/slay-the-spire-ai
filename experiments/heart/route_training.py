"""Learn route and campfire choices from paired branches of ordinary runs."""
import argparse
from pathlib import Path

import heart_native as engine
import numpy as np
import torch
from torch import nn

from .deck_training import funnel
from .heart_curriculum import read_model,save_model
from .recovery import panel
from .replay import apply_model
from .train import write


def collect(behavior,start,count,threads):
    w=apply_model("first",behavior)
    chunks=[]
    for offset in range(0,count,256):
        rows=np.asarray(engine.route_practice(w,start+offset,min(256,count-offset),threads),np.float32)
        rows[:,0]+=offset
        chunks.append(rows)
    return np.concatenate(chunks)


def samples(data):
    changes=np.any(data[1:,:2]!=data[:-1,:2],axis=1)
    starts=np.r_[0,np.flatnonzero(changes)+1]
    # Fit pairwise return differences. Each decision receives equal total
    # weight even when the map offers seven initial columns.
    left,right,weights=[],[],[]
    for start,end in zip(starts,np.r_[starts[1:],len(data)]):
        a,b=np.triu_indices(end-start,1)
        left.extend(start+a);right.extend(start+b);weights.extend([1/len(a)]*len(a))
    horizon,floors,acts,keys,heart=data[:,-5:].T
    value=.45*horizon+floors/13+acts+keys*.12+heart*8
    return (torch.from_numpy(data[:,3:-5]),torch.from_numpy(value),
            torch.tensor(left),torch.tensor(right),torch.tensor(weights,dtype=torch.float32),len(starts))


def fit(data,valid):
    x,y,left,right,weight,n=samples(data)
    vx,vy,vl,vr,vw,vn=samples(valid)
    torch.manual_seed(12431)
    mean=x.mean(0);scale=x.std(0).clamp_min(.12)
    x=(x-mean)/scale;vx=(vx-mean)/scale
    net=nn.Sequential(nn.Linear(x.shape[1],48),nn.ReLU(),nn.Linear(48,1,bias=False))
    opt=torch.optim.AdamW(net.parameters(),lr=.0015)
    best=float("inf")
    for epoch in range(1,33):
        for batch in torch.randperm(len(left)).split(1024):
            a,b=left[batch],right[batch]
            delta=(net(x[a])-net(x[b])).flatten()
            loss=((delta-y[a]+y[b]).square()*weight[batch]).mean()
            opt.zero_grad();loss.backward();opt.step()
        if epoch in (1,4,8,16,32):
            with torch.no_grad():
                pred=net(vx).flatten()
                err=float(((pred[vl]-pred[vr]-vy[vl]+vy[vr]).square()*vw).sum()/vw.sum())
                if err<best:
                    best=err
                    first=net[0].weight/scale
                    params=[first.tolist(),(net[0].bias-first@mean).tolist(),net[2].weight[0].tolist()]
                    stats=dict(epoch=epoch,heldout_pair_mse=err,train_choices=n,validation_choices=vn)
    return params,stats


def select(parent,params,start,threads):
    candidates=[]
    for scales in ([20,20],[80,80],[250,250],[160,40],[40,160]):
        model={**parent,"route":[*params,scales]}
        stats,r=panel(model,start,3072,threads)
        # Terminal encounters are sparse. Count the Act 3 boss and Act 4 as
        # additional progress once Heart counts tie.
        key=(len(stats["heart_encounter_seeds"]),stats["act4"],int((r[:,0]>=50).sum()),stats["act3"])
        candidates.append((key,model,dict(scales=scales,**stats,first_boss=int((r[:,0]>=50).sum()))))
    best=max(candidates,key=lambda c:c[0])
    return best[1],[c[2] for c in candidates]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=Path("build/route-policy"))
    p.add_argument("--threads",type=int,default=10)
    args=p.parse_args()
    args.out.mkdir(exist_ok=True,parents=True)
    torch.set_num_threads(3)
    parent=read_model("build/scenarios/challenger.json.gz")
    data=collect(parent,124000000,8192,args.threads)
    valid=collect(parent,125000000,1024,args.threads)
    params,one=fit(data,valid)
    first,choices=select(parent,params,126000000,args.threads)
    print("round one",one,first["route"][-1],flush=True)
    extra=collect(first,127000000,4096,args.threads)
    extra[:,0]+=8192
    params,two=fit(np.concatenate([data,extra]),valid)
    second,more=select(parent,params,128000000,args.threads)
    finalists=[]
    for name,model in (("first",first),("second",second)):
        summary,r=panel(model,129000000,8192,args.threads)
        finalists.append((len(summary["heart_encounter_seeds"]),summary["act4"],int((r[:,0]>=50).sum()),name,model,summary))
    best=max(finalists,key=lambda c:c[:3])
    model=best[4]
    save_model(args.out/"challenger.json.gz",model)
    report=dict(fits=[one,two],training_starts=[124000000,127000000],validation_start=125000000,
                scale_validation_starts=[126000000,128000000],rounds=[choices,more],finalist_start=129000000,
                finalist_results={row[3]:row[5] for row in finalists},selected=best[3],scales=model["route"][-1],test={})
    for start in (130000000,131000000):
        report["test"][start]={}
        for name,m in (("parent",parent),("route_and_rest",model)):
            s,r=panel(m,start,32768,args.threads)
            report["test"][start][name]={**s,**funnel(r)}
            print(start,name,report["test"][start][name],flush=True)
        write(args.out/"report.json",report)
    if any(sum(p["route_and_rest"][key] for p in report["test"].values())>
           1.10*sum(p["parent"][key] for p in report["test"].values()) for key in ("act3","first_boss")):
        report["components"]={}
        for i,name in enumerate(("map","rest")):
            scales=[0.,0.]
            scales[i]=model["route"][-1][i]
            variant={**model,"route":[*model["route"][:-1],scales]}
            summary,r=panel(variant,130000000,32768,args.threads)
            report["components"][name]={**summary,**funnel(r)}
        write(args.out/"report.json",report)


if __name__=="__main__":
    main()
