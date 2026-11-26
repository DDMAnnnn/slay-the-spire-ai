"""Fit a hand-aware late-act combat scorer from counterfactual fight outcomes."""
import argparse
import gzip
import json
from pathlib import Path

import heart_native as engine
import numpy as np
import torch
from torch import nn

from .boss_relics import compare
from .fit_combat import FightValue
from .replay import heart_encounters, load_model, set_late_combat
from .train import statistics, write


class HandValue(FightValue):
    def __init__(self):
        super().__init__()
        self.hand = nn.Embedding(engine.card_count+1, 16, padding_idx=0)
        nn.init.normal_(self.hand.weight, std=.08)
        with torch.no_grad():
            self.hand.weight[0].zero_()

    def forward(self, x):
        ids=x[:,4:14].long()
        hand=self.hand(ids).sum(dim=1)/(ids!=0).sum(dim=1,keepdim=True).clamp_min(1)
        hidden=self.card(x[:,2].long())+self.enemy(x[:,3].long())+hand+self.features(x[:,14:36])
        return self.readout(hidden.relu()).squeeze(-1)

    def export(self):
        card, enemy, features, bias, output = super().export()
        return [card,self.hand.weight.detach().tolist(),enemy,features,bias,output]


def collect(checkpoint,start,n,threads):
    w=load_model("first",checkpoint)
    chunks=[]
    for offset in range(0,n,128):
        rows=np.asarray(engine.late_combat_practice(w,start+offset,min(128,n-offset),threads),dtype=np.float32)
        if len(rows):
            rows[:,0]+=offset
            chunks.append(rows)
    return np.concatenate(chunks)


def groups(rows):
    starts=np.r_[0,np.flatnonzero(np.any(rows[1:,:2]!=rows[:-1,:2],axis=1))+1]
    return starts,np.diff(np.r_[starts,len(rows)])


def one_move(rows, predictions, scale):
    # All alternatives in each group use the same state and continuation policy.
    starts,counts=groups(rows)
    score=rows[:,36]+scale*predictions
    best=np.repeat(np.maximum.reduceat(score,starts),counts)
    index=np.minimum.reduceat(np.where(score==best,np.arange(len(rows)),len(rows)),starts)
    base=np.minimum.reduceat(np.where(rows[:,36]==0,np.arange(len(rows)),len(rows)),starts)
    advantage=rows[index,-1]-rows[base,-1]
    return dict(positions=len(starts),changed=int(np.sum(index!=base)),mean_gain=float(advantage.mean()),
                better=int((advantage>0).sum()),worse=int((advantage<0).sum()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference",type=Path,default=Path("build/first-boss-relics/challenger.json.gz"))
    p.add_argument("--draft",type=Path,default=Path("build/context-draft/id-only.json.gz"))
    p.add_argument("--out",type=Path,default=Path("build/late-combat-hand"))
    p.add_argument("--reuse-samples",action="store_true",help="reuse samples from the same checkpoints and seeds")
    p.add_argument("--threads",type=int,default=10)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    data={}
    for name,model,start,n in (("reference",args.reference,72000000,8192),
                               ("draft",args.draft,73000000,8192),("valid",args.reference,74000000,1024)):
        path=args.out/(name+".npy")
        if args.reuse_samples:
            data[name]=np.load(path)
        else:
            data[name]=collect(model,start,n,args.threads)
            np.save(path,data[name])
    train=np.concatenate((data["reference"],data["draft"]))
    valid=data["valid"]
    starts,counts=groups(train)
    y=torch.from_numpy((train[:,-1]-np.repeat(np.add.reduceat(train[:,-1],starts)/counts,counts)).astype("float32"))
    x,xv=torch.from_numpy(train),torch.from_numpy(valid)
    torch.manual_seed(72817)
    torch.set_num_threads(3)
    model=HandValue()
    opt=torch.optim.AdamW(model.parameters(),lr=.003)
    best=-np.inf
    for epoch in range(16):
        for idx in torch.randperm(len(x)).split(2048):
            loss=nn.functional.smooth_l1_loss(model(x[idx]),y[idx],beta=3.)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            predicted=model(xv).numpy()
        gain=one_move(valid,predicted,5)["mean_gain"]
        if gain>best:
            best=gain
            head=model.export()
            paired={s:one_move(valid,predicted,s) for s in (1,5,15)}
    report=dict(training_rows=len(train),training_positions=len(starts),
                training_by_act={str(a):int((train[starts,37]==a).sum()) for a in (2,3,4)},
                paired_validation=paired,validation_start=75000000,test_start=76000000)
    w=load_model("first",args.reference)
    trials={}
    for scale in (1,5,15):
        set_late_combat([*head,scale])
        trials[scale]=statistics(np.asarray(engine.batch(w,75000000,8192,args.threads)))
    selected=max(trials,key=lambda s:(trials[s]["act3"],trials[s]["mean_floor"]))
    report.update(validation=trials,selected_scale=selected,test={})
    with gzip.open(args.reference,"rt",encoding="utf8") as f:
        checkpoint=json.load(f)
    checkpoint["late_combat"]=[*head,selected]
    with gzip.open(args.out/"challenger.json.gz","wt",encoding="utf8") as f:
        json.dump(checkpoint,f)
    print("Paired",paired,"selected",selected,flush=True)
    for name,head in (("previous",None),("specialist",checkpoint["late_combat"])):
        set_late_combat(head)
        rows=np.asarray(engine.batch(w,76000000,32768,args.threads))
        if name=="previous":
            baseline=rows
        result=dict(**statistics(rows),**compare(baseline,rows),heart_encounter_seeds=heart_encounters(w,76000000,rows))
        report["test"][name]=result
        print(name,result,flush=True)
    write(args.out/"report.json",report)


if __name__=="__main__":
    main()
