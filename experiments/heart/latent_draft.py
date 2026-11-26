"""Use winning final deck clusters as training targets for an online mixture of draft preferences."""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .human_menus import TOKENS
from .human_strategy import load_data
from .train import write


def cluster(decks,groups):
    train=decks["train"].float()
    idf=(1.+len(train)/(1.+(train>0).sum(0))).log()
    features={k:F.normalize(v.float().log1p()*idf,dim=1) for k,v in decks.items()}
    x=features["train"]
    if groups==1:return {k:torch.zeros(len(v),dtype=torch.long,device=v.device) for k,v in features.items()},[len(x)]
    torch.manual_seed(27532+groups)
    centers=x[torch.randperm(len(x),device=x.device)[:groups]]
    for _ in range(60):
        idx=(x@centers.T).argmax(1)
        updated=torch.stack([x[idx==k].mean(0) if (idx==k).any() else centers[k] for k in range(groups)])
        updated=F.normalize(updated,dim=1)
        if (updated-centers).abs().max()<1.e-5:break
        centers=updated
    labels={k:(v@centers.T).argmax(1) for k,v in features.items()}
    return labels,torch.bincount(labels["train"],minlength=groups).tolist()


class Plans(nn.Module):
    def __init__(self,k):
        super().__init__()
        self.gate=nn.Parameter(torch.zeros(TOKENS,k));self.prior=nn.Parameter(torch.zeros(3,k))
        self.item=nn.Parameter(torch.zeros(k,TOKENS))
    def forward(self,ctx,act,ids,base):
        den=ctx.sum(1,keepdim=True).clamp_min(1).sqrt()
        log_prior=(ctx@self.gate/den+self.prior[act]).log_softmax(1)
        expert=(base[:,None,:]+self.item[:,ids.clamp_min(0)].permute(1,0,2)).log_softmax(2)
        mixed=torch.logsumexp(log_prior[:,:,None]+expert,dim=1)
        return mixed,expert,log_prior


def main():
    out=Path("build/latent-draft");torch.set_num_threads(2)
    device="cuda" if torch.cuda.is_available() else "cpu"
    source=Path("build/human-strategy");sets={};decks={};owners={}
    baseline=json.loads((source/"fitted.json").read_text())["context"]["config"]
    c,k,bias,up,repeat=[torch.tensor(v,device=device) for v in baseline]
    for s in ("train","validation","test","streamers"):
        raw=load_data(source/f"{s}.npz",device);ctx=raw["context"].float();act=raw["act"]
        opts=raw["options"];ids=opts[:,:,0].long();ix=ids.clamp_min(0)
        h=(ctx@c/ctx.sum(1,keepdim=True).clamp_min(1).sqrt()).tanh()
        base=(bias[act[:,None],ix]+up[ix]*opts[:,:,1]+repeat[ix]*ctx.gather(1,ix).log1p()
              +(h[:,None,:]*k[ix]).sum(2)).masked_fill(ids<0,-1.e8)
        sets[s]=ctx,act,ids,base,raw["pick"],raw["kind"]
        if s!="streamers":
            with np.load(out/f"{s}.npz") as v:
                decks[s]=torch.tensor(v["decks"],device=device)
                owners[s]=torch.tensor(v["owner"],device=device)
    def evaluate(s,model=None):
        ctx,act,ids,base,y,kind=sets[s]
        with torch.no_grad():
            z=base.log_softmax(1) if model is None else torch.cat([
                model(ctx[ix],act[ix],ids[ix],base[ix])[0] for ix in torch.arange(len(y),device=device).split(512)])
            loss=-z.gather(1,y[:,None]).squeeze(1)
            return {name:dict(n=int((kind==i).sum()),nll=float(loss[kind==i].mean()),
                              accuracy=float((z.argmax(1)[kind==i]==y[kind==i]).float().mean()))
                    for i,name in enumerate(("cards","relics"))}
    report={"base":{s:evaluate(s) for s in ("validation","test","streamers")},"fits":{}}
    for groups in (1,4,8,16):
        labels,clusters=cluster(decks,groups);goal=labels["train"][owners["train"]]
        model=Plans(groups).to(device);opt=torch.optim.AdamW(model.parameters(),lr=.018,weight_decay=.06)
        ctx,act,ids,base,y,kind=sets["train"];best=float("inf");history=[]
        torch.manual_seed(27547)
        for epoch in range(1,19):
            for ix in torch.randperm(len(y),device=device).split(256):
                mix,expert,prior=model(ctx[ix],act[ix],ids[ix],base[ix]);row=torch.arange(len(ix),device=device)
                loss=-.65*mix[row,y[ix]].mean()-.35*expert[row,goal[ix],y[ix]].mean()-.2*prior[row,goal[ix]].mean()
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            if epoch in (1,2,4,8,12,18):
                v=evaluate("validation",model);score=v["cards"]["nll"]+.15*v["relics"]["nll"]
                history.append(dict(epoch=epoch,validation=v))
                if score<best:
                    chosen=epoch;best=score;torch.save(model.state_dict(),out/f"plan-{groups}.pt")
        model.load_state_dict(torch.load(out/f"plan-{groups}.pt",map_location=device,weights_only=True))
        report["fits"][str(groups)]=dict(epoch=chosen,validation_loss=best,clusters=clusters,history=history,
                                         test=evaluate("test",model),streamers=evaluate("streamers",model))
        weights={k:p.detach().cpu().tolist() for k,p in model.named_parameters()}
        write(out/f"plan-{groups}.json",weights);write(out/"fit.json",report)
        print(groups,chosen,report["fits"][str(groups)]["test"],flush=True)
    report["selected"]=min(("4","8","16"),key=lambda k:report["fits"][k]["validation_loss"])
    write(out/"fit.json",report)


if __name__=="__main__":main()
