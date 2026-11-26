"""Learn corrections to public action scores from groups of matched stochastic continuations."""
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


class Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.item=nn.Embedding(engine.curriculum_items,20)
        self.foe=nn.Embedding(engine.curriculum_foes,20)
        self.layer=nn.Linear(58,20)
        self.out=nn.Linear(20,1,bias=False)
        nn.init.normal_(self.item.weight,std=.1)
        nn.init.normal_(self.foe.weight,std=.1)
        nn.init.zeros_(self.out.weight)

    def forward(self,rows):
        z=self.layer(rows[:,7:])+self.item(rows[:,1].long())+self.foe(rows[:,2].long())
        return rows[:,3]+8*(self.out(z.relu()).flatten()/8).tanh()

    def export(self,scale=1.):
        return [p.detach().tolist() for p in (self.item.weight,self.foe.weight,self.layer.weight,
                                              self.layer.bias,self.out.weight[0])]+[scale]


def make_banks(w,rows,start,limits,threads):
    rng=np.random.default_rng(start)
    banks={}
    for floor,limit in limits.items():
        available=start+np.flatnonzero(rows[:,0]>=floor)
        seeds=rng.choice(available,min(len(available),limit),replace=False)
        if len(seeds):
            banks[floor]=engine.Bank(w,seeds.tolist(),floor,threads)
    return banks


def update(net,optimizer,groups):
    rows=torch.from_numpy(np.concatenate(groups))
    starts=torch.from_numpy(np.r_[0,1+np.flatnonzero(np.diff(rows[:,0].numpy()))])
    lengths=torch.diff(torch.cat([starts,torch.tensor([len(rows)])]))
    max_error=0.
    for epoch in range(4):
        for batch_index,ids in enumerate(torch.randperm(len(starts)).split(384)):
            begin=starts[ids]
            offset=torch.arange(int(lengths[ids].max()))
            mask=offset[None,:]<lengths[ids,None]
            batch=rows[(begin[:,None]+offset)[mask]]
            logits=torch.full(mask.shape,-1e9)
            logits[mask]=net(batch)
            selected=torch.zeros_like(mask)
            selected[mask]=batch[:,4].bool()
            assert (selected.sum(1)==1).all()
            logp=logits.log_softmax(1)[selected]
            diff=logp-rows[begin,5]
            if epoch==0 and batch_index==0:
                max_error=float(diff.detach().abs().max())
            advantage=rows[begin,6]
            ratio=diff.exp()
            objective=torch.minimum(ratio*advantage,ratio.clamp(.8,1.2)*advantage)
            loss=-objective.mean()+.02*(ratio-1-diff).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),1.)
            optimizer.step()
    return dict(decisions=len(starts),rows=len(rows),initial_logp_error=max_error)


def score(stats,late):
    # Conditional entries supplement full starts when encounters are sparse.
    # Smooth all rates so a single Act 4 draw cannot determine the ranking.
    return (np.log((stats["act3"]+12)/stats["runs"])+
            .6*np.log((late["first_boss_clears"]+9)/(late["first_boss_entries"]+9))+
            .6*np.log((late["act4"]+5)/(late["first_boss_entries"]+5))+
            .4*np.log((late["shield_clears"]+3)/(late["shield_entries"]+3)))


def validate(parent,net,scale,banks,start,n,threads):
    model={**parent,"curriculum_actor":net.export(scale)}
    w=apply_model("first",model)
    boss=np.asarray(banks[50].progress(w,threads));shield=np.asarray(banks[55].progress(w,threads))
    late=dict(first_boss_entries=len(boss),first_boss_clears=int((boss[:,0]>=51).sum()),
              act4=int((boss[:,1]>=4).sum()),shield_entries=len(shield),shield_clears=int(shield[:,2].sum()))
    stats,raw=panel(model,start,n,threads)
    return model,{**stats,**funnel(raw),"late":late,"rank":float(score(stats,late)),"scale":scale}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=Path("build/matched-curriculum"))
    p.add_argument("--threads",type=int,default=10)
    p.add_argument("--updates",type=int,default=20)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(3);torch.manual_seed(160129)
    parent=read_model("build/trajectory-teacher/challenger.json.gz")
    w=apply_model("first",parent)
    scout=Path("build/ironclad-curriculum/parent-floors.npy")
    if not scout.exists():
        scout.parent.mkdir(exist_ok=True,parents=True)
        np.save(scout,np.asarray(engine.batch(w,159000000,32768,args.threads)))
    train=np.load(scout)
    valid=np.asarray(engine.batch(w,160000000,32768,args.threads))
    banks=make_banks(w,train,159000000,{1:480,8:480,16:480,23:480,33:512,40:384,50:256,55:64},args.threads)
    holdout=make_banks(w,valid,160000000,{50:160,55:64},args.threads)
    report=dict(training_start=159000000,validation_bank_start=160000000,screen_start=161000000,
                bank_sizes={k:len(b) for k,b in banks.items()},updates=[],validation=[],test={})
    print("training banks",report["bank_sizes"],flush=True)
    net=Residual()
    optimizer=torch.optim.AdamW(net.parameters(),lr=.0017)
    candidates=[]
    for iteration in range(1,args.updates+1):
        engine.set_curriculum(*net.export())
        chunks=[];stats={}
        for idx,(floor,bank) in enumerate(banks.items()):
            # Every trajectory in a group starts with the identical hidden draw and entry HP.
            rows,counts=engine.curriculum_practice(bank,w,0x82230+iteration*3373+idx*83,
                         48 if floor!=55 else 24,6,8 if iteration%2 else 16,args.threads)
            stats[floor]=counts
            if len(rows):
                rows[:,0]+=idx*100000
                chunks.append(rows)
        fit=update(net,optimizer,chunks)
        if fit["initial_logp_error"]>.0002:
            raise RuntimeError(f"native and torch policy differ: {fit}")
        rec=dict(iteration=iteration,training_counts=stats,**fit)
        report["updates"].append(rec)
        print("update",iteration,fit,"sampled Heart entries",sum(c[3] for c in stats.values()),flush=True)
        if iteration in (1,4,8,12,args.updates):
            for scale in (.5,1.5,4.):
                model,res=validate(parent,net,scale,holdout,161000000,1536,args.threads)
                res["iteration"]=iteration
                report["validation"].append(res);candidates.append((res["rank"],model,res))
            print("validation",iteration,max(v["rank"] for v in report["validation"]),flush=True)
        write(args.out/"report.json",report)
    finalists=[]
    for _,model,old in sorted(candidates,key=lambda entry:entry[0],reverse=True)[:4]:
        s,r=panel(model,162000000,8192,args.threads)
        rank=(len(s["heart_encounter_seeds"]),s["act4"],int((r[:,0]>=51).sum()),s["act3"])
        finalists.append((rank,model,dict(iteration=old["iteration"],scale=old["scale"],**s)))
    _,best,selected=max(finalists,key=lambda f:f[0])
    report["selection"]=[f[2] for f in finalists]
    report["selected"]=selected
    save_model(args.out/"challenger.json.gz",best)
    print("selected",selected,flush=True)
    for start in (163000000,164000000):
        report["test"][start]={}
        for name,m in (("parent",parent),("curriculum",best)):
            s,r=panel(m,start,32768,args.threads)
            report["test"][start][name]={**s,**funnel(r)}
            np.save(args.out/f"{name}-{start}.npy",r)
            print(start,name,report["test"][start][name],flush=True)
        write(args.out/"report.json",report)


if __name__=="__main__":
    main()
