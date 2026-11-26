"""Estimate next-act survival and reference deck composition from observed inventories."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

import heart_native as engine
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .heart_curriculum import read_model, save_model
from .replay import apply_model
from .train import write


def examples(out,threads):
    w=apply_model("first",read_model("build/stratified-gate/challenger.json.gz"))
    for offset in range(0,32768,1024):
        f=out/f"local-{offset}.jsonl.gz"
        if not f.exists():
            tmp=f.with_suffix(".tmp")
            with gzip.open(tmp,"wt") as sink:
                for row in engine.collect_deck_outcomes(w,232000000+offset,1024,threads):
                    row[0]+=232000000+offset
                    sink.write(json.dumps(row,separators=(",",":"))+"\n")
            tmp.replace(f)
        with gzip.open(f,"rt") as src:
            for row in map(json.loads,src):
                yield 0,row
    with gzip.open("build/public-expert-traces/replays.jsonl.gz","rt") as src:
        labels={r["seed"]:[r["act2"],r["act3"],r["act4"]] for r in map(json.loads,src) if not r["error"]}
    seen=set()
    with gzip.open("build/resource-policy/menus.jsonl.gz","rt") as src:
        for seed,_,kind,base,inv,_,_ in map(json.loads,src):
            hp,maximum,gold,floor,act=base[:5]
            key=seed,floor
            if act>3 or kind>1 or key in seen:continue
            seen.add(key)
            yield 1,[seed,int(labels[seed][act-1]),act,floor,kind,hp,maximum,gold,*[a-9 for a in inv]]


class InventoryForecast(nn.Module):
    def __init__(self,tokens):
        super().__init__()
        self.items=nn.Parameter(torch.randn(tokens,24)*.03)
        self.stage=nn.Parameter(torch.zeros(3,24))
        self.direct=nn.Parameter(torch.zeros(3,tokens,3))
        self.readout=nn.Parameter(torch.randn(3,24,3)*.04)
        self.bias=nn.Parameter(torch.zeros(3,3))
        self.context=nn.Parameter(torch.zeros(3,6,2))

    def forward(self,inv,act,context):
        h=torch.tanh(inv@self.items+self.stage[act])
        logit=(h.unsqueeze(2)*self.readout[act]).sum(1)+self.bias[act]
        # Keep public non-inventory covariates out of the domain discriminator.
        logit=logit+torch.stack([F.linear(inv,self.direct[k].T) for k in range(3)],dim=1)[torch.arange(len(inv)),act]
        adjustment=(context.unsqueeze(2)*self.context[act]).sum(1)
        return logit+F.pad(adjustment,(0,1))

    def export(self):
        return [p.detach().cpu().tolist() for p in (self.items,self.stage,self.direct,self.readout)]


def prepare(all_rows):
    tokens=engine.card_count+engine.relic_count
    n=len(all_rows)
    inv=np.zeros((n,tokens),dtype=np.uint8)
    other=[];ctx=[];counts=defaultdict(lambda:np.zeros(2,int))
    for i,(source,r) in enumerate(all_rows):
        seed,cleared,act,floor,shop,hp,maximum,gold,*items=r
        if min(items)<0 or max(items)>=tokens:raise ValueError("unrecognized inventory")
        inv[i]=np.bincount(items,minlength=tokens)
        # Mix the seed bits before splitting, and keep every floor of a run together.
        import hashlib
        bucket=int.from_bytes(hashlib.blake2b(str(seed).encode(),digest_size=4).digest(),"big")%10
        split=0 if bucket<8 else bucket-7
        phase=(max(floor-1,0)%17)//4
        other.append((source,cleared,act-1,phase,shop,split))
        ctx.append((hp/max(1,maximum),(maximum-75)/50,min(gold/200,3),phase/4,shop,1))
        counts[split,act,phase,shop][source]+=1
    # Give the two players the same act, floor band and menu-type frequencies.
    weights=[min(counts[s,a+1,p,k])/counts[s,a+1,p,k][d] for d,y,a,p,k,s in other]
    return (torch.from_numpy(inv),torch.tensor(other),torch.tensor(ctx,dtype=torch.float32),
            torch.tensor(weights,dtype=torch.float32))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=Path("build/deck-outcomes"))
    p.add_argument("--threads",type=int,default=4)
    args=p.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(23201);torch.set_num_threads(2)
    inv,records,ctx,weight=prepare(list(examples(out,args.threads)))
    domain,target,act,_,_,split=records.T
    model=InventoryForecast(inv.shape[1])
    opt=torch.optim.AdamW(model.parameters(),lr=.005,weight_decay=.012)
    indices=[torch.where((split==s)&(weight>0))[0] for s in range(3)]
    def losses(ix):
        w=weight[ix]
        z=model(inv[ix].float(),act[ix],ctx[ix])
        selected=z[torch.arange(len(ix)),domain[ix]]
        y=target[ix].float()
        outcome=F.binary_cross_entropy_with_logits(selected,y,reduction="none")
        origin=F.binary_cross_entropy_with_logits(z[:,2],domain[ix].float(),reduction="none")
        return outcome,origin,w,z
    @torch.no_grad()
    def assess(ix):
        pieces=[losses(i)[:3] for i in ix.split(512)]
        u,v,w=[torch.cat(parts) for parts in zip(*pieces)]
        return dict(n=len(ix),local=int((domain[ix]==0).sum()),expert=int((domain[ix]==1).sum()),
            survival_logloss=float((u*w).sum()/w.sum()),origin_logloss=float((v*w).sum()/w.sum()),
            local_logloss=float(u[domain[ix]==0].mean()),expert_logloss=float(u[domain[ix]==1].mean()))
    history=[];best=float("inf")
    for epoch in range(1,19):
        # Sampling inversely to domain prevalence prevents common early local games
        # from overwhelming the rarer later-act and expert supervision.
        idx=indices[0]
        draws=idx[torch.multinomial(weight[idx],min(32768,len(idx)),replacement=True)]
        for ix in draws.split(192):
            u,v,_,_=losses(ix)
            loss=u.mean()+.5*v.mean()+.001*model.direct.square().sum(1).mean()
            opt.zero_grad();loss.backward();opt.step()
        metric=assess(indices[1])
        history.append(dict(epoch=epoch,**metric))
        score=metric["survival_logloss"]+.5*metric["origin_logloss"]
        if score<best:
            best=score;chosen=epoch;torch.save(model.state_dict(),out/"model.pt")
    model.load_state_dict(torch.load(out/"model.pt",weights_only=True));model.eval()
    # Context-only comparator: an act/domain-specific smoothed survival probability.
    ref=torch.zeros((3,2))
    for a in range(3):
        for d in range(2):
            ix=indices[0][(act[indices[0]]==a)&(domain[indices[0]]==d)]
            ref[a,d]=(target[ix].float().sum()+1)/(len(ix)+2)
    ix=indices[2];w=weight[ix]
    base=F.binary_cross_entropy(ref[act[ix],domain[ix]],target[ix].float(),reduction="none")
    details=dict(selected_epoch=chosen,history=history,training=len(indices[0]),heldout=assess(ix),
        heldout_stage_baseline=float((base*w).sum()/w.sum()),heldout_domain_baseline=float(np.log(2)),
        own_scout_runs=32768,public_source="SnyderConsulting/sts_ml verified game traces")
    write(out/"fit.json",details)
    parent=read_model("build/stratified-gate/challenger.json.gz")
    save_model(out/"fitted.json.gz",{**parent,"deck_outcome":[*model.export(),[0.,0.,0.]]})
    print("held out",details["heldout"],"stage baseline",details["heldout_stage_baseline"],flush=True)


if __name__=="__main__":
    main()
