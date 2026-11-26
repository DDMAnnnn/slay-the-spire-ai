"""Learn joint resource decisions and test the frozen policy on ordinary runs."""
import argparse
import gzip
import json
import subprocess
from pathlib import Path

import heart_native as engine
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .heart_curriculum import read_model, save_model
from .inventory_search import panel_chunks
from .replay import apply_model
from .train import write


def features(base,options,inv):
    hp,maximum,gold,floor,act,red,green,blue,deck,up,strikes,defends,pots,slots=base
    missing=1-hp/max(1,maximum)
    local_floor=max(0,floor-1)%17
    common=[1.,missing,missing**2,(maximum-75)/50,min(gold/200,3),float(act==2),float(act>=3),
            local_floor/15,float(local_floor>=13),red,green,blue,min(deck/25,3),up/max(1,deck),
            strikes/5,defends/4,pots/max(1,slots),missing*float(local_floor>=13),float(act>=3 and not red)]
    return [common+[price/150,price/max(1,gold),upgrade,min(3,inv.count(token))/3]
            for token,_,upgrade,price in options]


def pack(rows,tokens):
    count=len(rows);width=max(len(x[5]) for x in rows)
    kinds=np.empty(count,np.int64);contexts=np.zeros((count,tokens),np.float32)
    opts=np.zeros((count,width),np.int64);cats=opts.copy();labels=np.empty(count,np.int64)
    x=np.zeros((count,width,23),np.float32);valid=np.zeros((count,width),bool)
    for i,(_,_,kind,base,inv,legal,label) in enumerate(rows):
        kinds[i],labels[i]=kind,label
        contexts[i]=np.bincount(inv,minlength=tokens)
        n=len(legal);opts[i,:n]=[a[0] for a in legal];cats[i,:n]=[a[1] for a in legal]
        x[i,:n]=features(base,legal,inv);valid[i,:n]=True
    return tuple(torch.from_numpy(v) for v in (kinds,contexts,opts,cats,x,valid,labels))


class ResourcePolicy(nn.Module):
    def __init__(self,tokens):
        super().__init__()
        self.category=nn.Parameter(torch.zeros(4,6,23))
        self.item=nn.Parameter(torch.zeros(4,tokens,23))
        self.context=nn.Parameter(torch.randn(tokens,12)*.08)
        self.option=nn.Parameter(torch.randn(4,tokens,12)*.08)

    def forward(self,kind,inventory,option,category,features,legal):
        ctx=F.dropout(inventory,.12,training=self.training)
        hidden=torch.tanh((ctx@self.context)/ctx.sum(1,keepdim=True).clamp_min(1).sqrt())
        k=kind[:,None]
        utility=((self.category[k,category]+self.item[k,option])*features).sum(-1)
        utility+=(hidden[:,None,:]*self.option[k,option]).sum(-1)
        return utility.masked_fill(~legal,-1e8)

    def export(self):
        return [p.detach().tolist() for p in (self.category,self.item,self.context,self.option)]


@torch.no_grad()
def assess(net,data):
    net.eval(); scores=[];labels=[];kind=[]
    for i in range(0,len(data[0]),256):
        k,c,o,t,x,m,y=(v[i:i+256] for v in data)
        z=net(k,c,o,t,x,m)
        scores.extend(F.cross_entropy(z,y,reduction="none").tolist())
        labels.extend(z.argmax(1).eq(y).tolist());kind.extend(k.tolist())
    k=np.array(kind)
    return [dict(count=int((k==a).sum()),nll=float(np.mean(np.asarray(scores)[k==a])),
                 accuracy=float(np.mean(np.asarray(labels)[k==a]))) for a in range(4)]


def fit(out):
    sizes=json.loads((out/"sizes.json").read_text())
    tokens=9+sizes["cards"]+sizes["relics"]+sizes["potions"]
    with gzip.open(out/"menus.jsonl.gz","rt") as f:
        rows=list(map(json.loads,f))
    splits=[pack([r for r in rows if keep(r[1])],tokens)
            for keep in (lambda x:x<8,lambda x:x==8,lambda x:x==9)]
    torch.manual_seed(22701);torch.set_num_threads(2)
    frequency=nn.Embedding(4,tokens)
    nn.init.zeros_(frequency.weight)
    opt0=torch.optim.AdamW(frequency.parameters(),lr=.04)
    for _ in range(10):
        for ix in torch.randperm(len(splits[0][0])).split(192):
            k,_,o,_,_,m,y=(v[ix] for v in splits[0])
            loss=F.cross_entropy(frequency(k).gather(1,o).masked_fill(~m,-1e8),y)
            opt0.zero_grad();loss.backward();opt0.step()
    class Frequency(nn.Module):
        def forward(self,k,c,o,t,x,m):
            return frequency(k).gather(1,o).masked_fill(~m,-1e8)
    baseline=assess(Frequency(),splits[2])
    model=ResourcePolicy(tokens)
    opt=torch.optim.AdamW(model.parameters(),lr=.009,weight_decay=.02)
    history=[];best=float("inf")
    for epoch in range(1,26):
        model.train()
        for ix in torch.randperm(len(splits[0][0])).split(128):
            k,c,o,t,x,m,y=(v[ix] for v in splits[0])
            loss=F.cross_entropy(model(k,c,o,t,x,m),y)+.01*model.item.square().sum(-1).mean()
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step()
        if epoch in (1,2,4,8,12,18,25):
            metrics=assess(model,splits[1]);score=np.mean([v["nll"] for v in metrics])
            history.append(dict(epoch=epoch,validation=metrics))
            if score<best:
                best=score;selected=epoch;torch.save(model.state_dict(),out/"resources.pt")
    model.load_state_dict(torch.load(out/"resources.pt",weights_only=True))
    return model.export(),dict(training_menus=len(splits[0][0]),selection=history,chosen_epoch=selected,
                               test_menus=assess(model,splits[2]),frequency_test=baseline,screens=["camp","shop","upgrade","remove"])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=Path("build/resource-policy"))
    p.add_argument("--threads",type=int,default=4)
    args=p.parse_args();out=args.out;path=out/"report.json"
    out.mkdir(parents=True,exist_ok=True)
    if not (out/"menus.jsonl.gz").exists():
        import sts_env as sts
        from .expert_draft import Menus
        cards=Menus().cards
        write(out/"sizes.json",dict(cards=engine.card_count,relics=engine.relic_count,potions=len(sts.POTION_NAMES),
                                    strikes=cards["Strike_R"],defends=cards["Defend_R"]))
        local=Path("build/public-expert-traces/python313/python.exe")
        cmd=[str(local.resolve())] if local.exists() else ["py","-3.13"]
        subprocess.run([*cmd,str(Path(__file__).with_name("resource_examples.py")),"--out",str(out)],check=True)
    parent=read_model("build/stratified-gate/challenger.json.gz")
    if (out/"fitted.json.gz").exists():
        fit_model=read_model(out/"fitted.json.gz")
        result=json.loads(path.read_text())
    else:
        params,details=fit(out)
        fit_model={**parent,"resource_policy":[*params,[1.]*4]}
        save_model(out/"fitted.json.gz",fit_model)
        result=dict(imitation=details,screen={},test={});write(path,result)
    params=fit_model["resource_policy"][:-1]
    def model(values):
        return {**parent,"resource_policy":[*params,values]}
    def fast(start,n,m):
        w=apply_model("first",m)
        x=np.asarray(engine.batch_seeds(w,list(range(start,start+n)),args.threads))
        return dict(n=n,entries=int(x[:,6].sum()),wins=int(x[:,4].sum()),act3=int((x[:,1]>=3).sum()),
                    boss=int((x[:,0]>=50).sum()),second=int((x[:,0]>=51).sum()),act4=int((x[:,1]>=4).sum()))
    for weight in (0.,.35,.7,1.):
        k=str(weight)
        if k not in result["screen"]:
            result["screen"][k]=fast(227000000,4096,model([weight]*4))
            write(path,result)
    def rank(s):return (s["entries"],s["act4"],s["second"],s["boss"],s["act3"])
    strength=max((.35,.7,1.),key=lambda s:rank(result["screen"][str(s)]))
    result["strength"]=strength
    candidate=model([strength]*4)
    save_model(out/"challenger.json.gz",candidate)
    arms={"parent":parent,"joint":candidate}
    for k,name in enumerate(("camp","shop","upgrade","remove")):
        mix=[0.]*4;mix[k]=strength;arms[name]=model(mix)
    for name,m in arms.items():
        if name in result["test"]: continue
        w=apply_model("first",m)
        result["test"][name]=panel_chunks(out,name,w,228000000,32768,args.threads)
        write(path,result)
        print(name,result["test"][name],flush=True)
        if name=="joint" and len(result["test"][name]["heart_encounter_seeds"])<len(result["test"]["parent"]["heart_encounter_seeds"]):
            result["individual_test_reason"]="Identify which resource decisions reduced ordinary reach."
            write(path,result)


if __name__=="__main__":
    main()
