"""Fit a common candidate scorer across the reference player's noncombat decisions."""
import base64
import gzip
import hashlib
import json
from pathlib import Path

import heart_native as engine
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .heart_curriculum import read_model, save_model
from .replay import apply_model, rows
from .train import write

SIZES=json.loads(Path("build/resource-policy/sizes.json").read_text())
C,R,P=(SIZES[k] for k in ("cards","relics","potions"))
BASE=16+C+R+P
NEVENT=len(read_model("build/route-irl/selected-whole_map.json.gz")["events"])
NEOW=BASE+8+16*NEVENT
TOKENS=NEOW+256
TASKS=10
H=16
D=17


def represent(o,legal,q):
    screen=int(o[6]);event=int(o[9])-1;task=int(o[26])
    kind={1:0 if event==5 else 1,2:2,3:3,8:4,7:5,4:6 if task==4 else 7 if task==3 else 8,6:9}.get(screen)
    if kind is None:return
    cards=[15+int(c[0]) for c in rows(o,"deck") if c[0]]
    relics=[15+C+int(c[0]) for c in rows(o,"relics") if c[0]]
    pots=[15+C+R+int(p) for p in o[17:22] if p]
    bag=cards+relics+pots
    legal=[a for a in legal if sts_choose<=a<sts_choose+64]
    if len(legal)<2:return
    top=max(q[a] for a in legal)
    miss=1-o[0]/max(1,o[1])
    common=[1.,miss,miss**2,(o[1]-75)/50,min(3,o[2]/200),o[4]==2,o[4]>=3,
            max(0,int(o[3])-1)%17/15,*o[12:15],min(o[22]/25.,3)]
    choices=rows(o,"choices");items={}
    for a in legal:
        ix=a-sts_choose;typ,item,up,extra=choices[ix,:4].astype(int)
        if typ==1:token=15+item
        elif typ==2:token=15+C+item
        elif typ==3:token=15+C+R+item
        elif typ==4:token=1
        elif typ==5:token=2 if item==1 else 3
        elif typ==6:token=0
        elif typ==8:token=BASE+8+16*event+ix
        elif typ==9:token=NEOW+8*(item-1)+(up-1)
        elif typ==10:token=BASE+item
        elif typ==11:token=6
        elif typ==12:token=5
        elif typ==14:token=4
        else:return
        if token<0 or token>=TOKENS:return
        price=extra if screen==8 else 0
        features=[*common,up>0 and typ==1,price/150.,price/max(1,o[2]),min(3,bag.count(token))/3.,
                  np.clip((q[a]-top)/12.,-8.,0.)]
        items[a]=(token,typ,features)
    return kind,bag,items

sts_choose=81


class CandidatePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.context=nn.Embedding(TOKENS,H)
        self.action=nn.Embedding(TOKENS,H)
        nn.init.normal_(self.context.weight,std=.07);nn.init.normal_(self.action.weight,std=.07)
        self.item=nn.Parameter(torch.zeros(TOKENS,D))
        self.category=nn.Parameter(torch.zeros(TASKS,15,D))
        self.task_item=nn.Parameter(torch.zeros(TASKS,TOKENS))
        self.task_hidden=nn.Parameter(torch.zeros(TASKS,H))

    def forward(self,kind,bag,tok,typ,x,legal):
        inv=bag>=0
        context=(self.context(bag.clamp_min(0))*inv[:,:,None]).sum(1)/inv.sum(1).clamp_min(1).sqrt()[:,None]
        h=torch.tanh(context+self.task_hidden[kind])
        score=(x*(self.item[tok]+self.category[kind[:,None],typ])).sum(2)
        score+=self.task_item[kind[:,None],tok]+(self.action(tok)*h[:,None,:]).sum(2)
        return score.masked_fill(~legal,-1e8)

    def export(self):
        return [p.detach().cpu().tolist() for p in (self.context.weight,self.action.weight,self.item,self.category,self.task_item,self.task_hidden)]


def dataset(out):
    path=out/"data.npz"
    if path.exists():
        with np.load(path) as data:return [torch.from_numpy(data[k]) for k in ("kind","bag","tok","typ","x","legal","y","split")]
    model=read_model("build/route-irl/selected-whole_map.json.gz")
    w=apply_model("first",model);samples=[];subsets={}
    with gzip.open(out/"choices.jsonl.gz","rt") as fp:
        for seed,packet,legal,picked in map(json.loads,fp):
            o=np.frombuffer(base64.b64decode(packet),dtype="<i2").astype(np.float32)
            mask=np.zeros(146,np.uint8);mask[legal]=1;q=np.asarray(engine.scores(w,o,mask))
            encoded=represent(o,legal,q)
            if encoded is None:continue
            kind,bag,items=encoded
            if picked not in items:continue
            # Indistinguishable copies compete as one item, including their parent score.
            choices=[];index={}
            for a,(t,typ,x) in items.items():
                key=(t,typ,tuple(x));index[a]=choices.index(key) if key in choices else len(choices)
                if index[a]==len(choices):choices.append(key)
            if len(choices)<2:continue
            if seed not in subsets:subsets[seed]=int.from_bytes(hashlib.sha256(str(seed).encode()).digest()[:4],"big")%10
            samples.append((kind,bag,choices,index[picked],subsets[seed]))
    n=len(samples);mw=max(len(v[2]) for v in samples);mb=max(len(v[1]) for v in samples)
    kind=np.zeros(n,np.int64);bag=np.full((n,mb),-1,np.int16);tok=np.zeros((n,mw),np.int16);typ=np.zeros_like(tok)
    x=np.zeros((n,mw,D),np.float32);legal=np.zeros((n,mw),bool);y=kind.copy();split=kind.copy()
    for i,(k,inv,choices,pick,bucket) in enumerate(samples):
        kind[i]=k;bag[i,:len(inv)]=inv
        for j,(t,c,features) in enumerate(choices):tok[i,j]=t;typ[i,j]=c;x[i,j]=features;legal[i,j]=True
        y[i]=pick;split[i]=bucket
    np.savez_compressed(path,kind=kind,bag=bag,tok=tok,typ=typ,x=x,legal=legal,y=y,split=split)
    return [torch.from_numpy(v) for v in (kind,bag,tok,typ,x,legal,y,split)]


def main():
    out=Path("build/strategic-apprentice")
    torch.manual_seed(26227);torch.set_num_threads(2)
    device="cuda" if torch.cuda.is_available() else "cpu"
    kind,bag,tok,typ,x,legal,y,split=[v.to(device) for v in dataset(out)]
    tr,va,te=[torch.where(s)[0] for s in (split<8,split==8,split==9)]
    net=CandidatePolicy().to(device)
    opt=torch.optim.AdamW(net.parameters(),lr=.009,weight_decay=.035)
    counts=torch.bincount(kind[tr],minlength=TASKS).float().clamp_min(1)
    weight=counts.rsqrt()
    def forward(ix):return net(kind[ix],bag[ix].long(),tok[ix].long(),typ[ix].long(),x[ix],legal[ix])
    @torch.no_grad()
    def assess(ix):
        z=torch.cat([forward(b) for b in ix.split(256)]);loss=F.cross_entropy(z,y[ix],reduction="none")
        pred=z.argmax(1);parent=x[ix,:,-1].masked_fill(~legal[ix],-1e8).argmax(1)
        stats={}
        for k in range(TASKS):
            select=kind[ix]==k
            if select.any():
                stats[k]=dict(n=int(select.sum()),nll=float(loss[select].mean()),accuracy=float((pred[select]==y[ix][select]).float().mean()),
                              parent_accuracy=float((parent[select]==y[ix][select]).float().mean()))
        return stats
    best=float("inf");history=[]
    for ep in range(1,19):
        for ids in tr[torch.randperm(len(tr),device=device)].split(192):
            l=F.cross_entropy(forward(ids),y[ids],reduction="none");w=weight[kind[ids]]
            loss=(l*w).sum()/w.sum();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2.);opt.step()
        if ep in (1,2,4,8,12,18):
            stats=assess(va);value=np.mean([v["nll"] for v in stats.values()]);history.append(dict(epoch=ep,loss=value))
            if value<best:
                selected=ep;best=value;torch.save(net.state_dict(),out/"chosen.pt")
    net.load_state_dict(torch.load(out/"chosen.pt",map_location=device,weights_only=True))
    base=read_model("build/route-irl/selected-whole_map.json.gz")
    save_model(out/"apprentice.json.gz",{**base,"strategic_apprentice":[*net.export(),[1.]*TASKS]})
    result=dict(splits=[len(s) for s in (tr,va,te)],validation=history,selected=selected,test=assess(te))
    write(out/"fit.json",result);print(result,flush=True)


if __name__=="__main__":main()
