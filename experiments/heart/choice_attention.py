"""Compare a public action-set attention critic with a per-action critic on identical fight labels."""
import argparse
from pathlib import Path

import heart_native as engine
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .deck_training import funnel
from .heart_curriculum import read_model,save_model
from .recovery import panel
from .train import write


class Groups:
    def __init__(self,files,device):
        chunks,starts,sizes=[],[],[]
        offset=0
        for path in files:
            with np.load(path) as data:
                r=data["rows"]
            local=np.r_[0,1+np.flatnonzero(np.any(r[1:,:2]!=r[:-1,:2],axis=1))]
            starts.extend(offset+local);sizes.extend(np.diff(np.r_[local,len(r)]))
            chunks.append(r[:,3:].copy());offset+=len(r)
        self.rows=torch.as_tensor(np.concatenate(chunks),device=device)
        self.starts=torch.tensor(starts,device=device)
        self.sizes=torch.tensor(sizes,device=device)
        self.slots=torch.arange(max(sizes),device=device)

    def batch(self,indices):
        lengths=self.sizes[indices]
        slots=self.slots[:int(lengths.max())]
        mask=slots[None,:]<lengths[:,None]
        rows=self.rows[(self.starts[indices,None]+slots[None,:]).masked_fill(~mask,0)]
        y=rows[:,:,2:6].clone()
        y[:,:,1]/=80.
        return rows,mask,y


class ActionSet(nn.Module):
    def __init__(self,attention):
        super().__init__()
        self.attention=attention
        self.items=nn.Embedding(engine.curriculum_items,24)
        self.foes=nn.Embedding(engine.curriculum_foes,24)
        self.input=nn.Linear(58,24)
        self.query=nn.Linear(24,8,bias=False)
        self.key=nn.Linear(24,8,bias=False)
        self.value=nn.Linear(24,8,bias=False)
        self.mix=nn.Linear(8,24,bias=False)
        self.heads=nn.Linear(24,4)
        nn.init.normal_(self.items.weight,std=.15)
        nn.init.normal_(self.foes.weight,std=.15)

    def forward(self,rows,mask):
        h=(self.input(rows[:,:,8:])+self.items(rows[:,:,0].long())+self.foes(rows[:,:,1].long())).relu()
        if self.attention:
            scores=self.query(h)@self.key(h).transpose(-1,-2)/8**.5
            weights=scores.masked_fill(~mask[:,None,:],-1e9).softmax(-1)
            h=(h+self.mix(weights@self.value(h))).relu()
        return self.heads(h)

    def export(self,utility):
        with torch.no_grad():
            out=self.heads.weight.T@torch.as_tensor(utility,device=self.heads.weight.device)
            mix=self.mix.weight if self.attention else torch.zeros_like(self.mix.weight)
            return [x.cpu().tolist() for x in (self.items.weight,self.foes.weight,self.input.weight,
                    self.input.bias,self.query.weight,self.key.weight,self.value.weight,mix,out)]


def train_model(train,valid,attention,device):
    torch.manual_seed(18411)
    model=ActionSet(attention).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0012,weight_decay=.01)
    utilities=([6.,6.,-2.,-.8],[4.,9.,-1.,-1.6])
    sel=torch.tensor(utilities[0],device=device)
    history=[];best=float("inf")
    for epoch in range(1,33):
        for ids in torch.randperm(len(train.starts),device=device).split(512):
            rows,mask,labels=train.batch(ids)
            pred=model(rows,mask)
            diff=pred-pred[:,:1]
            target=labels-labels[:,:1]
            # Every state gets one vote; a hand with many targets need not dominate the loss.
            pair_mask=mask.clone();pair_mask[:,0]=False
            pair=F.smooth_l1_loss(diff,target,reduction="none",beta=.2).mean(-1)
            absolute=F.smooth_l1_loss(pred,labels,reduction="none",beta=.2).mean(-1)
            loss=((pair*pair_mask).sum(1)/pair_mask.sum(1)+.25*(absolute*mask).sum(1)/mask.sum(1)).mean()
            optimizer.zero_grad();loss.backward();optimizer.step()
        if epoch in (1,4,8,16,32):
            total=torch.zeros(4,device=device);zero=torch.zeros_like(total)
            count=0;utility_mse=0.
            with torch.no_grad():
                for ids in torch.arange(len(valid.starts),device=device).split(512):
                    rows,mask,labels=valid.batch(ids)
                    pred=model(rows,mask)
                    target=labels-labels[:,:1]
                    errors=pred-pred[:,:1]-target
                    mask[:,0]=False
                    e=errors[mask];y=target[mask]
                    total+=e.square().sum(0);zero+=y.square().sum(0)
                    utility_mse+=float((e@sel).square().sum());count+=len(y)
            score=utility_mse/count
            history.append(dict(epoch=epoch,pairs=count,paired_mse=(total/count).tolist(),
                                zero_mse=(zero/count).tolist(),utility_mse=score))
            if score<best:
                best=score;chosen=epoch
                configs=[model.export(v) for v in utilities]
    return configs,chosen,history,utilities


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=Path("build/choice-attention"))
    p.add_argument("--threads",type=int,default=8)
    p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(3)
    def files(which):
        seeds=(176,180) if which=="train" else (177,181)
        return sorted(Path("build/combat-returns").glob(f"{seeds[0]}*.npz"))+sorted(Path("build/fatal-combat").glob(f"{seeds[1]}*.npz"))
    train=Groups(files("train"),args.device)
    valid=Groups(files("valid"),args.device)
    parent=read_model("build/trajectory-teacher/challenger.json.gz")
    report=dict(train_states=len(train.starts),valid_states=len(valid.starts),methods={},test={})
    candidates={}
    for attention in (False,True):
        name="attention" if attention else "per_action"
        configs,epoch,history,utilities=train_model(train,valid,attention,args.device)
        method=dict(epoch=epoch,history=history,validation=[])
        options=[]
        for params,utility in zip(configs,utilities):
            for scale in (1.,4.,12.):
                policy={**parent,"choice_attention":[*params,scale]}
                s,r=panel(policy,184000000,3072,args.threads)
                result={**s,**funnel(r),"scale":scale,"utility":utility}
                method["validation"].append(result)
                rank=result["first_boss"]+2*result["second_boss"]+.1*result["act3"]+4*result["act4"]
                options.append((rank,policy,result))
        _,candidates[name],method["selected"]=max(options,key=lambda x:x[0])
        report["methods"][name]=method
        print(name,method["selected"],flush=True)
        save_model(args.out/(name+".json.gz"),candidates[name])
    for name,model in {"parent":parent,**candidates}.items():
        s,r=panel(model,185000000,32768,args.threads)
        report["test"][name]={**s,**funnel(r)}
        print(name,report["test"][name],flush=True)
        write(args.out/"report.json",report)


if __name__=="__main__":
    main()
