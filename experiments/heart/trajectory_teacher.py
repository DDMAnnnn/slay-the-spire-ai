"""Distill offline fight trajectories into a residual public-observation policy."""
import argparse
import json
from pathlib import Path

import heart_native as engine
import numpy as np
import torch

from .deck_training import funnel
from .direct_actor import Actor, Decisions, padded
from .heart_curriculum import read_model, save_model
from .recovery import panel
from .replay import apply_model
from .train import write


def collect(parent,start,count,paths,threads):
    w=apply_model("first",parent)
    chunks=[]
    total=np.zeros(5,dtype=np.int64)
    for i in range(0,count,32):
        raw,stats=engine.trajectory_practice(w,start+i,min(32,count-i),paths,threads)
        total+=stats
        if raw:
            x=np.asarray(raw,dtype=np.float32)
            x[:,0]+=i
            chunks.append(x)
    return np.concatenate(chunks),dict(zip(
        ("fights","parent_clears","teacher_clears","improved_fights","decisions"),map(int,total)))


def fit(train,valid):
    torch.manual_seed(32418)
    net=Actor()
    optimizer=torch.optim.AdamW(net.parameters(),lr=.003)
    train=Decisions(train)
    valid=Decisions(valid)
    snapshots=[]
    for epoch in range(1,25):
        for ix in torch.randperm(len(train.starts)).split(384):
            mask,rows=train.batch(ix)
            # The policy only needs to learn the correction to its existing preference.
            scores=padded(mask,net(rows)+rows[:,55]/12.,-1e6)
            labels=padded(mask,rows[:,54],0)
            importance=(1+padded(mask,rows[:,56],0).amax(1).clamp(0,40)/20)
            loss=(-(scores.log_softmax(1)*labels).sum(1)*importance).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch in (2,6,12,24):
            with torch.no_grad():
                mask,rows=valid.batch(torch.arange(len(valid.starts)))
                base=padded(mask,rows[:,55],-1e6)
                scores=padded(mask,net(rows)+rows[:,55]/12.,-1e6)
                labels=padded(mask,rows[:,54],0)
                hit=labels.gather(1,scores.argmax(1,keepdim=True)).flatten()
                old=labels.gather(1,base.argmax(1,keepdim=True)).flatten()
                snapshots.append((net.export(),dict(epoch=epoch,teacher_agreement=float(hit.mean()),
                        parent_agreement=float(old.mean()),corrected=int(((hit>0)&(old==0)).sum()),
                        lost=int(((hit==0)&(old>0)).sum()))))
    return snapshots


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=Path("build/trajectory-teacher"))
    p.add_argument("--threads",type=int,default=10)
    p.add_argument("--reuse",action="store_true")
    args=p.parse_args()
    args.out.mkdir(exist_ok=True,parents=True)
    torch.set_num_threads(3)
    parent=read_model("build/route-policy/challenger.json.gz")
    report=dict(paths=24,data={},validation=[],test={})
    data={}
    for name,start,count in (("train",138000000,2048),("valid",139000000,256)):
        f=args.out/(name+".npz")
        if args.reuse:
            with np.load(f) as archive:
                data[name],info=archive["rows"],json.loads(str(archive["description"]))
        else:
            data[name],info=collect(parent,start,count,report["paths"],args.threads)
            np.savez_compressed(f,rows=data[name],description=json.dumps(info))
        report["data"][name]=dict(start=start,runs=count,**info)
        print(name,info,flush=True)
    candidates=[]
    for head,accuracy in fit(data["train"],data["valid"]):
        for scale in (4,12,24):
            m={**parent,"trajectory_teacher":[head,scale]}
            s,r=panel(m,140000000,2048,args.threads)
            record={**accuracy,"scale":scale,**s,**funnel(r)}
            report["validation"].append(record)
            key=(s["act4"],record["first_boss"],s["act3"],s["mean_floor"])
            candidates.append((key,m,record))
    _,model,rec=max(candidates,key=lambda row:row[0])
    print("selected",rec,flush=True)
    report["selected"]=rec
    save_model(args.out/"challenger.json.gz",model)
    for start in (141000000,142000000):
        report["test"][start]={}
        for name,policy in (("parent",parent),("student",model)):
            s,r=panel(policy,start,32768,args.threads)
            report["test"][start][name]={**s,**funnel(r)}
            print(start,name,report["test"][start][name],flush=True)
        write(args.out/"report.json",report)


if __name__=="__main__":
    main()
