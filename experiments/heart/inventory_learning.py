"""Learn the marginal combat value of an item in the inventory it will join."""
import argparse
import json
from pathlib import Path

import heart_native as engine
import numpy as np
import sts_env as sts
import torch
from torch import nn

from .boss_draft import rank
from .deck_training import funnel
from .heart_curriculum import read_model,save_model
from .inventory_benchmark import winners
from .recovery import panel
from .replay import apply_model
from .train import write


def natural_profiles(w,start,rows,threads):
    rng=np.random.default_rng(start)
    result=[]
    for floor in (8,15,23,32,40,49):
        seeds=start+np.flatnonzero(rows[:,0]>=floor)
        use=rng.choice(seeds,min(400,len(seeds)),replace=False)
        result.extend(engine.bank_inventories(engine.Bank(w,use.tolist(),floor,threads)))
    return result


def families(own,human,n,seed):
    rng=np.random.default_rng(seed)
    upgradeable={c//256-1 for deck,_ in [*own,*human] for c in deck if c%256}
    regular=upgradeable-{sts.CARD_IDS.index(n) for n in ("Strike_R","Defend_R","Bash","AscendersBane")}
    def additions(source,allow_up=True):
        return [c if allow_up else (c//256)*256 for c in source if c//256-1 in regular]
    rows=[]
    for _ in range(n):
        od,orr=own[rng.integers(len(own))]
        hd,hr=human[rng.integers(len(human))]
        if rng.random()<.3:
            deck=list(hd);relics=list(hr)
        else:
            deck=list(od);relics=list(orr)
        if rng.random()<.65:
            source=additions(hd)
            share=rng.uniform(0,.65)
            deck.extend(c for c in source if rng.random()<share)
            relics.extend(r for r in hr if r not in relics and rng.random()<share)
        if rng.random()<.3:
            _,rr=own[rng.integers(len(own))]
            relics=list(rr)
        deck=deck[:78];relics=relics[:37]
        rows.append((deck,relics))
        source,other=human[rng.integers(len(human))]
        add=additions(source) or list(source)
        card=int(rng.choice(add))
        rows.append((deck+[card],relics))
        j=rng.integers(len(deck))
        edited=deck.copy()
        if not (edited[j]%256) and edited[j]//256-1 in upgradeable and rng.random()<.5:
            edited[j]+=1
        elif len(edited)>1:
            edited.pop(j)
        rows.append((edited,relics))
        candidates=[r for r in other if r not in relics]
        changed=relics.copy()
        if candidates: changed.append(int(rng.choice(candidates)))
        elif changed:changed.pop(rng.integers(len(changed)))
        rows.append((deck,changed))
        # Bundles expose interactions that one-card perturbations alone rarely visit.
        bundle=rng.choice(add,size=min(len(add),int(rng.integers(2,6))),replace=False).tolist()
        rows.append((deck+bundle,changed))
    return rows


def encode(rows):
    x=np.zeros((len(rows),2*engine.card_count+engine.relic_count),np.float32)
    for i,(cards,relics) in enumerate(rows):
        d=np.asarray(cards)
        np.add.at(x[i],d//256-1+(d%256)*engine.card_count,1)
        np.add.at(x[i],2*engine.card_count+np.asarray(relics,dtype=int)-1,1)
    return x


def dataset(own,human,count,seed,w,threads):
    profiles=families(own,human,count,seed)
    outcomes=[]
    for i in range(0,len(profiles),500):
        ys=engine.grade_inventories(w,profiles[i:i+500],seed+i//5,5,3,threads)
        outcomes.append(np.asarray(ys,np.float32))
    return encode(profiles).reshape(count,5,-1),np.concatenate(outcomes).reshape(count,5,4)


def predictions(net,x):
    return net(x).sigmoid()


def fit(train,valid):
    torch.manual_seed(16633)
    torch.set_num_threads(3)
    x,y=map(torch.from_numpy,train)
    vx,vy=map(torch.from_numpy,valid)
    net=nn.Sequential(nn.Linear(x.shape[-1],48),nn.Tanh(),nn.Linear(48,4))
    optimizer=torch.optim.AdamW(net.parameters(),lr=.0015,weight_decay=.012)
    best=float("inf")
    snapshots=[]
    for epoch in range(1,49):
        for ids in torch.randperm(len(x)).split(192):
            pred=predictions(net,x[ids]);labels=y[ids]
            delta=pred[:,1:]-pred[:,:1]
            actual=labels[:,1:]-labels[:,:1]
            loss=(pred-labels).square().mean()+2*(delta-actual).square().mean()
            optimizer.zero_grad();loss.backward();optimizer.step()
        if epoch in (1,4,12,24,48):
            with torch.no_grad():
                pred=predictions(net,vx)
                expected=pred[:,1:]-pred[:,:1]
                actual=vy[:,1:]-vy[:,:1]
                mse=float((pred-vy).square().mean())
                dmse=float((expected-actual).square().mean())
                corr=np.corrcoef(expected.flatten(),actual.flatten())[0,1]
                snapshots.append(dict(epoch=epoch,mse=mse,paired_mse=dmse,paired_correlation=float(corr),
                                     zero_difference_mse=float(actual.square().mean())))
                if mse+2*dmse<best:
                    best=mse+2*dmse
                    weights=[v.detach().tolist() for v in (net[0].weight.T,net[0].bias,net[2].weight,net[2].bias)]
                    chosen=epoch
    return weights,snapshots,chosen


def confirm(out,threads):
    parent=read_model("build/trajectory-teacher/challenger.json.gz")
    model=read_model(out/"challenger.json.gz")
    report=json.loads((out/"report.json").read_text(encoding="utf8"))
    for seed,names in ((171000000,("parent","inventory")),(169000000,("add","edit","relic"))):
        results=report["test"].setdefault(str(seed),{})
        for name in names:
            current=parent if name=="parent" else model
            if name in ("add","edit","relic"):
                params=model["inventory_value"]
                scales=[s if channel==name else 0. for channel,s in zip(("add","edit","relic"),params[-1])]
                current={**model,"inventory_value":[*params[:-1],[*scales,params[-1][3]]]}
            stats,rows=panel(current,seed,32768,threads)
            results[name]={**stats,**funnel(rows)}
            write(out/"channels.json",report)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=Path("build/inventory-learning"))
    p.add_argument("--threads",type=int,default=10)
    p.add_argument("--reuse",action="store_true")
    p.add_argument("--confirm",action="store_true",help="Run the independent confirmation and channel ablations")
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    if args.confirm:
        confirm(args.out,args.threads)
        return
    parent=read_model("build/trajectory-teacher/challenger.json.gz")
    w=apply_model("first",parent)
    people,_=winners(Path("build/expert-runs/runs"))
    order=np.random.default_rng(16631).permutation(len(people))
    subsets=([people[i] for i in order[::5]],[people[i] for j,i in enumerate(order) if j%5])
    data={};report=dict(data={},validation=[],selection=[],test={})
    for name,start,n,people_set in (("train",159000000,6144,subsets[1]),("valid",166000000,1024,subsets[0])):
        f=args.out/(name+".npz")
        if not args.reuse:
            scout=np.load("build/ironclad-curriculum/parent-floors.npy") if name=="train" else np.asarray(engine.batch(w,start,32768,args.threads))
            own=natural_profiles(w,start,scout,args.threads)
            x,y=dataset(own,people_set,n,start,w,args.threads)
            np.savez_compressed(f,x=x,y=y)
        else:
            with np.load(f) as z:x,y=z["x"],z["y"]
        data[name]=(x,y)
        report["data"][name]=dict(start=start,families=n,profiles=len(x)*5,standalone_fights=len(x)*5*4*3,
                                   human_profiles=len(people_set))
        print(name,report["data"][name],flush=True)
    params,history,epoch=fit(data["train"],data["valid"])
    report.update(fit=history,chosen_epoch=epoch)
    print("fit",history,flush=True)
    trials=[]
    for scales in ((30,30,30,.3),(90,90,90,.6),(200,200,200,1.),(500,500,500,1.),(200,50,500,.4),
                   (40,150,300,1.5),(350,180,60,.1),(100,300,120,2.)):
        model={**parent,"inventory_value":[*params,scales]}
        s,r=panel(model,167000000,3072,args.threads)
        result={**s,**funnel(r),"scales":scales}
        trials.append((model,result));report["validation"].append(result)
    options=[]
    for model,old in sorted(trials,key=lambda t:rank(t[1]),reverse=True)[:3]:
        s,r=panel(model,168000000,8192,args.threads)
        result={**s,**funnel(r),"scales":old["scales"]}
        report["selection"].append(result);options.append((model,result))
    chosen,selected=max(options,key=lambda t:rank(t[1]))
    report["scales"]=selected["scales"];save_model(args.out/"challenger.json.gz",chosen)
    print("selected",selected,flush=True)
    for start in (169000000,170000000):
        report["test"][start]={}
        for name,m in (("parent",parent),("inventory",chosen)):
            s,r=panel(m,start,32768,args.threads)
            report["test"][start][name]={**s,**funnel(r)}
            print(start,name,report["test"][start][name],flush=True)
        write(args.out/"report.json",report)


if __name__=="__main__":
    main()
