"""Fit room utilities by differentiating through soft path planning on each visible map."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .heart_curriculum import read_model, save_model
from .train import write


def features(c,row):
    act,hp,maximum,gold,pots,deck,red,green,blue=c
    miss=1-hp/max(1,maximum)
    return [1.,miss,miss**2,min(gold/200,3.),min(pots,4)/3.,min(deck,50)/24.,
            act==2,act==3,row/14.,not red,not green,not blue,miss*(act>=2)]


def load_data(out):
    result=[]
    with gzip.open(out/"routes.jsonl.gz","rt") as src:
        for seed,bucket,boards,path in map(json.loads,src):
            for act,row,ctx,key,options,pick in path:
                if len(options)<2:continue
                board=np.asarray(boards[str(act)],np.int16).reshape(15,7,2)
                elite=np.zeros((15,7),np.uint8)
                if not ctx[7] and len(key)==2 and 0<=key[0]<7 and 0<=key[1]<15:elite[key[1],key[0]]=1
                legal=np.zeros(7,bool);legal[options]=True
                result.append((board[:,:,0],board[:,:,1],elite,row,features(ctx,row),legal,pick,bucket))
    return [torch.from_numpy(np.asarray(values)) for values in zip(*result)]


class RoutePolicy(nn.Module):
    def __init__(self,graph):
        super().__init__()
        self.graph=graph
        self.reward=nn.Parameter(torch.zeros(13,10))  # nine room IDs and burning-elite marker
        self.depth=nn.Parameter(torch.zeros(10))
        self.edge=nn.Parameter(torch.zeros(2,9,9))  # constant and missing-HP terms
        self.branch=nn.Parameter(torch.zeros(3))

    def forward(self,room,bits,key,row,x,legal):
        n=len(x);x=x.float();room=room.long();bits=bits.long()
        ix=torch.arange(n,device=x.device)
        u=x@self.reward
        y=torch.arange(15,device=x.device).view(1,15,1)/14
        immediate=u[ix[:,None,None],room]+y*self.depth[room]+key*(u[:,9,None,None]+y*self.depth[9])
        if not self.graph:return immediate[ix,row].masked_fill(~legal,-1e9)
        q=immediate[:,14];chosen=torch.zeros_like(q)
        branch=self.branch[(x[:,6]+2*x[:,7]).long()]
        for yy in range(14,-1,-1):
            if yy<14:
                children=torch.stack([F.pad(q[:,:-1],(1,0)),q,F.pad(q[:,1:],(0,1))],dim=2)
                above=room[:,yy+1]
                dest=torch.stack([F.pad(above[:,:-1],(1,0)),above,F.pad(above[:,1:],(0,1))],dim=2)
                src=room[:,yy,:,None]
                children=children+self.edge[0,src,dest]+x[:,1,None,None]*self.edge[1,src,dest]
                edges=(bits[:,yy,:,None]&bits.new_tensor([1,2,4]))!=0
                count=edges.sum(2);ln=count.clamp_min(1).float().log()
                value=torch.logsumexp(children.masked_fill(~edges,-1e6),2)-ln+branch[:,None]*ln
                q=immediate[:,yy]+.9*torch.where(count>0,value,0.)
            chosen=torch.where((row==yy)[:,None],q,chosen)
        return chosen.masked_fill(~legal,-1e9)

    def export(self):
        return [p.detach().cpu().tolist() for p in (self.reward,self.depth,self.edge,self.branch)]+[self.graph]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=Path("build/route-irl"));args=p.parse_args();out=args.out
    device="cuda" if torch.cuda.is_available() else "cpu";torch.set_num_threads(2)
    parts=[x.to(device) for x in load_data(out)];bucket=parts.pop()
    indices=[torch.where(mask)[0] for mask in (bucket<8,bucket==8,bucket==9)]
    report=dict(counts=[len(i) for i in indices],models={})
    for graph,name in ((False,"next_room"),(True,"whole_map")):
        torch.manual_seed(24210);net=RoutePolicy(graph).to(device)
        opt=torch.optim.AdamW(net.parameters(),lr=.022,weight_decay=.04)
        def batch(ix):return net(*(p[ix] for p in parts[:-1]))
        @torch.no_grad()
        def assess(ix):
            z=torch.cat([batch(ids) for ids in ix.split(512)])
            y=parts[-1][ix]
            return dict(n=len(y),accuracy=float(z.argmax(1).eq(y).float().mean()),nll=float(F.cross_entropy(z,y)),
                        uniform_logloss=float(parts[-2][ix].sum(1).float().log().mean()))
        history=[];best=float("inf")
        for ep in range(1,25):
            for ids in indices[0][torch.randperm(len(indices[0]),device=device)].split(256):
                loss=F.cross_entropy(batch(ids),parts[-1][ids])+.003*net.edge.square().mean()
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2.);opt.step()
            if ep in (1,2,4,8,16,24):
                v=assess(indices[1]);history.append(dict(epoch=ep,**v))
                if v["nll"]<best:
                    best=v["nll"];chosen=ep;torch.save(net.state_dict(),out/f"{name}.pt")
        net.load_state_dict(torch.load(out/f"{name}.pt",map_location=device,weights_only=True))
        report["models"][name]=dict(validation=history,chosen_epoch=chosen,test=assess(indices[2]))
        m=read_model("build/stratified-gate/challenger.json.gz")
        save_model(out/f"{name}.json.gz",{**m,"map_policy":[*net.export(),1.]})
        print(name,report["models"][name]["test"],flush=True);write(out/"fit.json",report)


if __name__=="__main__":
    main()
