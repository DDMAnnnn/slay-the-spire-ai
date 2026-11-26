"""Fit masked choice models to recorded human menus, then test them in the simulator."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

import heart_native as engine
import numpy as np
import sts_env as sts
import torch

from .boss_draft import rank
from .deck_training import funnel
from .heart_curriculum import read_model, save_model
from .recovery import panel
from .train import write

DATA_REV="097aaf3564c2247835162d267cbc7c55d2c9039e"


def enum(header,name):
    s=Path(header).read_text(encoding="utf8")
    block=s.split("enum class "+name+" {",1)[1].split("};",1)[0]
    return [part.split("=")[0].strip() for part in block.split(",") if part.strip()]


class Menus:
    def __init__(self):
        self.cards={c:i for i,c in enumerate(sts.CARD_IDS)}
        names=Path("build/rl-fork/include/constants/Relics.h").read_text(encoding="utf8")
        ids=re.findall(r'"([^"]*)"',names.split("relicIds[]",1)[1].split("};",1)[0])
        assert len(ids)==engine.relic_count
        self.relics={s:i for i,s in enumerate(ids)}
        neow="build/rl-fork/include/game/Neow.h"
        self.bonuses=enum(neow,"Bonus")
        self.costs=enum(neow,"Drawback")
        assert (len(self.bonuses),len(self.costs))==tuple(engine.neow_shape)

    @staticmethod
    def card_feature(card,act,up):
        if card<0:
            return (0,0,0)
        n=engine.card_count
        return (1+card,1+n+act*n+card,1+4*n+card if up else 0)

    @staticmethod
    def relic_feature(relic,act):
        if relic<0:
            return (0,0,0)
        n=engine.relic_count
        return (1+relic,1+n+act*n+relic,0)

    def neow_feature(self,b,c):
        n=len(self.bonuses)
        k=len(self.costs)
        return (1+b,1+n+c,1+n+k+b*k+c)

    def examples(self,r,omitted):
        for offer in r["card_choices"]:
            floor=offer["floor"]
            if not 1<=floor<=51:
                continue
            act=int(floor>17)+int(floor>34)
            choices=list(dict.fromkeys([offer["picked"],*offer["not_picked"],"SKIP"]))
            menu=[]
            for name in choices:
                if name=="SKIP":
                    menu.append((0,0,0))
                else:
                    raw=name.split("+")[0]
                    if raw not in self.cards:
                        omitted["card:"+raw]+=1
                        break
                    menu.append(self.card_feature(self.cards[raw],act,"+" in name))
            else:
                if len(menu)>1:
                    yield "card",menu
        for act,offer in enumerate(r["boss_relics"][:2]):
            menu=[]
            for name in dict.fromkeys([offer["picked"],*offer["not_picked"],"SKIP"]):
                if name=="SKIP":
                    menu.append((0,0,0))
                elif name in self.relics:
                    menu.append(self.relic_feature(self.relics[name],act))
                else:
                    omitted["relic:"+name]+=1
                    break
            else:
                if len(menu)>1:
                    yield "relic",menu
        bonuses=[r.get("neow_bonus"),*(r.get("neow_bonuses_skipped_log") or [])]
        costs=[r.get("neow_cost"),*(r.get("neow_costs_skipped_log") or [])]
        if len(bonuses)!=4 or len(costs)!=4:
            omitted["incomplete opening"]+=1
            return
        menu=[]
        for b,c in zip(bonuses,costs):
            c="LOSE_STARTER_RELIC" if b=="BOSS_RELIC" else c
            if b not in self.bonuses or c not in self.costs:
                omitted["unknown opening"]+=1
                return
            menu.append(self.neow_feature(self.bonuses.index(b),self.costs.index(c)))
        yield "neow",menu


def collect(root,encoder):
    groups=[defaultdict(list),defaultdict(list)]
    games=[Counter(),Counter()]
    omitted=Counter()
    seen=set()
    # Exclude the deliberately constrained 'lose all gold' series.
    for sub in ("200-rotating-sample","chegs","panacea-ironclad-sample"):
        for path in sorted((root/sub).rglob("*.run")):
            r=json.loads(path.read_bytes())
            seed=str(r["seed_played"])
            if (r.get("character_chosen")!="IRONCLAD" or r.get("ascension_level")!=20 or seed in seen or
                any(r.get(k) for k in ("is_daily","chose_seed","is_trial","is_endless"))):
                continue
            seen.add(seed)
            valid=int(int.from_bytes(hashlib.sha256(seed.encode()).digest()[:4],"big")%5==0)
            games[valid][sub]+=1
            for tag,menu in encoder.examples(r,omitted):
                groups[valid][tag].append(menu)
    return groups,dict(source_commit=DATA_REV,games=[dict(g) for g in games],omitted=dict(omitted))


def pack(groups):
    length=max(map(len,groups))
    x=torch.zeros((len(groups),length,3),dtype=torch.long)
    mask=torch.zeros(x.shape[:2],dtype=torch.bool)
    for i,group in enumerate(groups):
        x[i,:len(group)]=torch.tensor(group)
        mask[i,:len(group)]=True
    return x,mask


def fit(train,valid,size,regularization):
    x,mask=pack(train)
    v,vmask=pack(valid)
    theta=torch.zeros(size,requires_grad=True)
    opt=torch.optim.LBFGS([theta],max_iter=90,line_search_fn="strong_wolfe")
    def scores(x,mask):
        return (theta[x]*(x!=0)).sum(2).masked_fill(~mask,-1e6)
    def loss():
        opt.zero_grad()
        err=-scores(x,mask).log_softmax(1)[:,0].sum()+regularization*theta.square().sum()/2
        err.backward()
        return err
    opt.step(loss)
    with torch.no_grad():
        pred=scores(v,vmask)
        best=pred==pred.max(1).values[:,None]
        # A tie gets fractional credit. Menus store the actual choice in column zero.
        acc=float((best[:,0]/best.sum(1)).mean())
        return theta.detach().numpy(),dict(train=len(x),valid=len(v),agreement=acc,
              random_agreement=float((1/vmask.sum(1)).mean()),logloss=float(-pred.log_softmax(1)[:,0].mean()))


def parameters(encoder,groups):
    dims=dict(card=1+5*engine.card_count,relic=1+3*engine.relic_count,
              neow=1+len(encoder.bonuses)+len(encoder.costs)+len(encoder.bonuses)*len(encoder.costs))
    weights={}
    report={}
    for name,size in dims.items():
        trials=[fit(groups[0][name],groups[1][name],size,r) for r in (1.,4.,16.)]
        i=min(range(3),key=lambda j:trials[j][1]["logloss"])
        weights[name],selected=trials[i]
        report[name]={**selected,"l2":(1,4,16)[i]}
    def total(name,indices):
        return float(weights[name][list(indices)].sum())
    cards=np.array([[[total("card",encoder.card_feature(i,act,u)) for u in (0,1)]
                     for i in range(engine.card_count)] for act in range(3)])
    relics=np.array([[total("relic",encoder.relic_feature(i,a)) for i in range(engine.relic_count)] for a in range(2)])
    neow=np.array([[total("neow",encoder.neow_feature(i,c)) for c in range(len(encoder.costs))]
                   for i in range(len(encoder.bonuses))])
    return (cards,relics,neow),report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,default=Path("build/expert-runs/runs"))
    p.add_argument("--out",type=Path,default=Path("build/expert-draft"))
    p.add_argument("--threads",type=int,default=10)
    args=p.parse_args()
    args.out.mkdir(exist_ok=True,parents=True)
    torch.set_num_threads(2)
    enc=Menus()
    groups,report=collect(args.source,enc)
    tables,report["fit"]=parameters(enc,groups)
    print("human menus",report,flush=True)
    parent=read_model("build/trajectory-teacher/challenger.json.gz")
    def test(scales,start,runs):
        policy={**parent,"expert_prior":[(t*s).tolist() for t,s in zip(tables,scales)]}
        s,r=panel(policy,start,runs,args.threads)
        return {**s,**funnel(r)},policy
    options=[]
    report["validation"]=[]
    for scales in ((3,3,1),(8,8,3),(20,15,3),(50,30,6),(100,60,6),(30,6,1),(12,25,1),(100,20,1)):
        stats,_=test(scales,149000000,3072)
        options.append((scales,stats))
        report["validation"].append({**stats,"scales":scales})
    report["selection"]=[]
    finalists=[]
    for mix,_ in sorted(options,key=lambda x:rank(x[1]),reverse=True)[:3]:
        s,_=test(mix,150000000,8192)
        finalists.append((mix,s))
        report["selection"].append({**s,"scales":mix})
    mix,_=max(finalists,key=lambda x:rank(x[1]))
    report.update(selected=mix,test={},components={})
    print("selected",mix,flush=True)
    for start in (151000000,152000000):
        s,r=panel(parent,start,32768,args.threads)
        t,model=test(mix,start,32768)
        if start==151000000:
            save_model(args.out/"challenger.json.gz",model)
        report["test"][start]={"parent":{**s,**funnel(r)},"expert":t}
        print(start,report["test"][start],flush=True)
        write(args.out/"report.json",report)
    for i,name in enumerate(("cards","boss_relics","opening")):
        subset=[0,0,0]
        subset[i]=mix[i]
        report["components"][name],_=test(subset,151000000,32768)
        print(name,report["components"][name],flush=True)
    write(args.out/"report.json",report)


if __name__=="__main__":
    main()
