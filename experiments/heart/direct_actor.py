"""Replace ordinary combat scoring with a policy trained on whole decisions."""
import argparse
from pathlib import Path

import heart_native as engine
import numpy as np
import torch
from torch import nn

from .heart_curriculum import read_model, save_model
from .late_combat import groups
from .replay import heart_encounters, load_model, set_actor
from .train import statistics, write


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.item=nn.Embedding(engine.actor_items,32)
        self.hand=nn.Embedding(engine.actor_items,32,padding_idx=0)
        self.target=nn.Embedding(engine.enemy_count,32)
        self.features=nn.Linear(40,32)
        self.output=nn.Linear(32,1,bias=False)
        for embedding in (self.item,self.hand,self.target):
            nn.init.normal_(embedding.weight,std=.08)
        with torch.no_grad():
            self.hand.weight[0].zero_()

    def forward(self,x):
        hand=x[:,4:14].long()
        context=self.hand(hand).sum(1)/(hand!=0).sum(1,keepdim=True).clamp_min(1)
        hidden=context+self.item(x[:,2].long())+self.target(x[:,3].long())+self.features(x[:,14:54])
        return self.output(hidden.relu()).squeeze(-1)

    def export(self):
        return [p.detach().tolist() for p in (self.item.weight,self.hand.weight,self.target.weight,
                                             self.features.weight,self.features.bias,self.output.weight[0])]


def collect(w,seed,n,threads):
    chunks=[]
    for offset in range(0,n,128):
        x=np.asarray(engine.actor_practice(w,seed+offset,min(128,n-offset),threads),dtype=np.float32)
        if len(x):
            x[:,0]+=offset
            chunks.append(x)
    return np.concatenate(chunks)


class Decisions:
    def __init__(self,rows):
        self.x=torch.from_numpy(rows)
        starts,lengths=groups(rows)
        self.starts=torch.from_numpy(starts)
        self.lengths=torch.from_numpy(lengths)
        self.labeled=np.add.reduceat(rows[:,54],starts)>0

    def batch(self,idx):
        lengths=self.lengths[idx]
        k=torch.arange(int(lengths.max()))
        mask=k[None,:]<lengths[:,None]
        rows=self.x[(self.starts[idx,None]+k)[mask]]
        return mask,rows


def padded(mask,values,fill):
    x=values.new_full(mask.shape,fill)
    x[mask]=values
    return x


def fit(data,validation,imitation,epochs=16,on_epoch=None):
    torch.manual_seed(8427)
    model=Actor()
    opt=torch.optim.AdamW(model.parameters(),lr=.004)
    train=Decisions(data)
    valid=Decisions(validation)
    best=-np.inf
    for epoch in range(epochs):
        for idx in torch.randperm(len(train.starts)).split(256):
            mask,rows=train.batch(idx)
            logits=padded(mask,model(rows),-1e6)
            teacher=padded(mask,rows[:,54],0)
            # One distribution per decision, so states with many targets have equal weight.
            # Include downstream progress to credit holding a potion across fights.
            returns=padded(mask,(rows[:,55]+.5*rows[:,56])/4,-1e6).softmax(1)
            targets=imitation*teacher+(1-imitation)*returns
            loss=-(targets*logits.log_softmax(1)).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            mask,rows=valid.batch(torch.arange(len(valid.starts)))
            pred=padded(mask,model(rows),-1e6).argmax(1,keepdim=True)
            agreement=float(padded(mask,rows[:,54],0).gather(1,pred).mean())
            outcome=rows[:,55]+.5*rows[:,56]
            chosen=padded(mask,outcome,0).gather(1,pred).squeeze(1)
            previous=padded(mask,outcome*rows[:,54],0).sum(1)
            gain=float((chosen-previous)[valid.labeled].mean())
            metric=agreement if imitation==1 else gain
        if on_epoch is not None:
            on_epoch(epoch+1,model,gain,agreement)
        if metric>best:
            best=metric
            params=model.export()
            details=dict(epoch=epoch+1,validation_positions=len(valid.starts),agreement=agreement,
                         single_move_gain=gain,training_positions=len(train.starts),training_rows=len(data))
    return params,details


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=Path("build/direct-actor"))
    parser.add_argument("--reference",type=Path,default=Path("build/first-boss-relics/challenger.json.gz"))
    parser.add_argument("--staged",type=Path,default=Path("build/heart-curriculum/staged.json.gz"))
    parser.add_argument("--reuse-samples",action="store_true",help="reuse matching local first-round samples")
    parser.add_argument("--threads",type=int,default=10)
    args=parser.parse_args()
    args.out.mkdir(exist_ok=True,parents=True)
    torch.set_num_threads(3)
    data={}
    for name,path,start,n in (("reference",args.reference,82000000,4096),("staged",args.staged,83000000,4096),
                              ("valid",args.reference,84000000,768)):
        w=load_model("first",path)
        file=args.out/(name+".npy")
        if args.reuse_samples:
            data[name]=np.load(file)
        else:
            data[name]=collect(w,start,n,args.threads)
            np.save(file,data[name])
    w=load_model("first",args.reference)
    train=np.concatenate((data["reference"],data["staged"]))
    variants={}
    report=dict(validation_start=85000000,test_start=87000000,fit={},validation={},test={})
    def trial(name,params,details):
        set_actor(params)
        variants[name]=params
        checkpoint=read_model(args.reference)
        checkpoint["direct_actor"]=params
        save_model(args.out/(name+".json.gz"),checkpoint)
        report["fit"][name]=details
        result=np.asarray(engine.batch(w,85000000,2048,args.threads))
        report["validation"][name]=statistics(result)
        print(name,details,report["validation"][name],flush=True)
    for name,mix in (("imitation",1.),("outcomes",.25)):
        trial(name,*fit(train,data["valid"],mix))
    def rank(name):
        result=report["validation"][name]
        return result["act4"],result["act3"],result["mean_floor"]
    first=max(variants,key=rank)
    set_actor(variants[first])
    # Aggregate the mistakes the new policy actually visits. Continue the
    # counterfactuals with that policy too, rather than training only on expert states.
    extra=collect(w,86000000,4096,args.threads)
    trial("second_round",*fit(np.concatenate((train,extra)),data["valid"],.25))
    selected=max(variants,key=rank)
    model=read_model(args.reference)
    model["direct_actor"]=variants[selected]
    save_model(args.out/"challenger.json.gz",model)
    report.update(selected=selected,collection_policy=first)
    for name,params in {"reference":None,**variants}.items():
        set_actor(params)
        result=np.asarray(engine.batch(w,87000000,32768,args.threads))
        summary=dict(**statistics(result),heart_encounter_seeds=heart_encounters(w,87000000,result))
        report["test"][name]=summary
        print(name,summary,flush=True)
        write(args.out/"report.json",report)


if __name__=="__main__":
    main()
