"""Learn fight survival and exit health conditional on enemy, deck and inventory."""
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .train import write


SIZES=json.loads(Path("build/resource-policy/sizes.json").read_text())
C,R,P=(SIZES[k] for k in ("cards","relics","potions"))
H=24


def tokens(row):
    return ([1+2*c+u for c,u in row["deck"]] + [1+2*C+r for r in row["relics"]]
            + [1+2*C+R+p for p in row["potions"]])


def context(row):
    hp=row["hp"]/max(1,row["max_hp"]); act=row["act"]
    return [1., hp, hp**2, (row["max_hp"]-75.)/40., len(row["deck"])/25.,
            len(row["relics"])/12., len(row["potions"])/2., min(1.,max(0., (row["floor"]-1-17*(act-1))/15.)),
            float(act==2),float(act==3),float(act==4)]


class Forecast(nn.Module):
    def __init__(self,encounters,inventory):
        super().__init__()
        self.inventory=inventory
        self.item=nn.Embedding(1+2*C+R+P,H,padding_idx=0)
        self.foe=nn.Embedding(encounters,H)
        self.ctx=nn.Linear(11,H,bias=False)
        self.readout=nn.Linear(H,2)
        nn.init.normal_(self.item.weight, std=.06)
        nn.init.zeros_(self.item.weight.data[0]);nn.init.normal_(self.foe.weight,std=.12)

    def forward(self,items,foes,x):
        h=self.ctx(x)+self.foe(foes)
        if self.inventory:
            h=h+self.item(items).sum(1)/((items!=0).sum(1).clamp_min(1).sqrt()[:,None])
        return self.readout(torch.tanh(h))

    def export(self):
        return [p.detach().cpu().tolist() for p in (self.item.weight,self.foe.weight,self.ctx.weight,
                                                     self.readout.weight,self.readout.bias)]


def main():
    out=Path("build/encounter-forecast")
    with gzip.open(out/"fights.jsonl.gz","rt") as fp: data=list(map(json.loads,fp))
    device="cuda" if torch.cuda.is_available() else "cpu";torch.set_num_threads(2)
    bags=[tokens(v) for v in data]; padded=np.zeros((len(bags),max(map(len,bags))),np.int64)
    for i,ids in enumerate(bags):padded[i,:len(ids)]=ids
    item=torch.tensor(padded,device=device)
    foe=torch.tensor([v["encounter"] for v in data],device=device)
    x=torch.tensor([context(v) for v in data],device=device)
    y=torch.tensor([[v["survived"],min(1.,v["exit_hp"]/max(1,v["max_hp"]))] for v in data],device=device)
    buckets=np.array([int.from_bytes(hashlib.sha256(str(v["seed"]).encode()).digest()[:4],"big")%10 for v in data])
    tr,va,te=[torch.tensor(np.flatnonzero(mask),device=device) for mask in (buckets<8,buckets==8,buckets==9)]
    result={"fights":[len(tr),len(va),len(te)],"games":[len({data[i]["seed"] for i in ids.tolist()}) for ids in (tr,va,te)],"models":{}}
    bank={}
    for a in range(1,4):
        src=[data[i] for i in tr.tolist() if data[i]["act"]==a and data[i]["room"]!=6]
        freq=np.bincount([v["encounter"] for v in src],minlength=int(foe.max())+1)
        # One shared encounter distribution per act, drawn entirely from training games.
        bank[str(a)]=[(int(i),float(freq[i]/freq.sum())) for i in np.flatnonzero(freq)]
    for name,inventory in (("context",False),("inventory",True)):
        torch.manual_seed(268775)
        model=Forecast(int(foe.max())+1,inventory).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.006,weight_decay=.025)
        @torch.no_grad()
        def assess(ix):
            z=torch.cat([model(item[b],foe[b],x[b]) for b in ix.split(512)])
            p=z.sigmoid(); truth=y[ix];live=truth[:,0]==1
            value=p[:,0]*(.55+.45*p[:,1])
            return dict(win_nll=float(F.binary_cross_entropy_with_logits(z[:,0],truth[:,0])),
                        win_brier=float(((p[:,0]-truth[:,0])**2).mean()),
                        survivor_hp_mae=float((p[live,1]-truth[live,1]).abs().mean()),
                        utility_mse=float(((value-truth[:,0]*(.55+.45*truth[:,1]))**2).mean()))
        best=float("inf");history=[]
        for epoch in range(1,33):
            for b in tr[torch.randperm(len(tr),device=device)].split(192):
                z=model(item[b],foe[b],x[b]);truth=y[b]
                loss=F.binary_cross_entropy_with_logits(z[:,0],truth[:,0])
                loss+=2.*((z[:,1].sigmoid()-truth[:,1]).square()*truth[:,0]).sum()/truth[:,0].sum().clamp_min(1.)
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
                optimizer.step()
            if epoch in (1,2,4,8,12,16,24,32):
                v=assess(va);history.append(dict(epoch=epoch,**v))
                score=v["win_nll"]+v["survivor_hp_mae"]
                if score<best:
                    best=score;selected=epoch;torch.save(model.state_dict(),out/f"{name}.pt")
        model.load_state_dict(torch.load(out/f"{name}.pt",map_location=device,weights_only=True))
        result["models"][name]=dict(epoch=selected,validation=history,test=assess(te))
        if inventory:
            write(out/"weights.json",dict(weights=model.export(),encounters=bank))
        write(out/"fit.json",result);print(name,result["models"][name]["test"],flush=True)


if __name__=="__main__":
    main()
