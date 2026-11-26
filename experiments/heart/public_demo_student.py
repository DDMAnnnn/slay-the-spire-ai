"""Learn combat action probabilities from independently replayed tactical demonstrations."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import heart_native as engine
import numpy as np
import sts_env as sts
import torch

from .heart_curriculum import read_model
from .replay import apply_model
from .shared_combat import demo_rows, interchangeable, projected
from .train import write
from .transfer_probe import EXAMPLE, evaluate, model_code


def batches(data, indices, device, size=256):
    for start in range(0,len(indices),size):
        rows = data[indices[start:start+size]]
        obs = torch.as_tensor(rows["obs"].astype(np.float32), device=device)
        legal = torch.zeros((len(rows),sts.N_ACTIONS),device=device,dtype=torch.bool)
        legal[:,:81] = torch.as_tensor(rows["legal"].astype(bool),device=device)
        action = torch.as_tensor(rows["action"].astype(np.int64),device=device)
        yield obs,legal,interchangeable(obs,legal,action)


def measure(net, data, device):
    result = np.zeros((5,3))
    net.eval()
    with torch.no_grad():
        for obs,mask,labels in batches(data,np.arange(len(data)),device):
            logp = net.dist(obs,mask).logits
            loss = -logp.masked_fill(~labels,-1e9).logsumexp(1)
            correct = labels.gather(1,logp.argmax(1,keepdim=True))[:,0]
            for act in range(5):
                keep = obs[:,4]==act if act else torch.ones_like(correct)
                result[act] += [int(keep.sum()),int(correct[keep].sum()),float(loss[keep].sum())]
    return [dict(count=int(n),accuracy=c/max(1,n),nll=s/max(1,n)) for n,c,s in result]


class Actor:
    def __init__(self,net,device,weights,threads,visible_intents=False):
        self.net,self.device,self.weights,self.threads=net,device,weights,threads
        self.blend=.5
        self.visible_intents=visible_intents

    @torch.inference_mode()
    def dist(self,obs,mask):
        raw = obs.cpu().numpy()
        legal = mask.cpu().numpy().astype(np.uint8)
        prior = torch.as_tensor(engine.prior_batch(self.weights,raw,legal,self.threads),device=self.device)
        # Follow-up screens have different actions and retain the original policy.
        use = mask[:,80] & (mask.sum(1)>1)
        if use.any():
            x = torch.as_tensor(projected(raw[use.cpu().numpy()],self.visible_intents),device=self.device)
            logp = self.net.dist(x,mask[use]).logits
            prior[use] = self.blend*prior[use]+logp
        return SimpleNamespace(logits=prior)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("build/public-expert-traces"))
    p.add_argument("--balance-acts",action="store_true")
    p.add_argument("--threads",type=int,default=2)
    p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    args=p.parse_args()
    root=args.root
    visible_intents="rendering" in json.loads((root/"layout.json").read_text())
    torch.manual_seed(225018)
    torch.set_num_threads(2)
    rng=np.random.default_rng(225018)
    data={k:demo_rows(root,k) for k in ("train","validation","test")}
    partitions=[set(x["seed"].tolist()) for x in data.values()]
    if any(partitions[i]&partitions[j] for i,j in ((0,1),(0,2),(1,2))):
        raise ValueError("training and evaluation runs overlap")
    net,_=model_code().load(EXAMPLE/"weights.pt",args.device)
    inherited=torch.load("build/ironclad-transfer/update-64.pt",map_location=args.device,weights_only=True)
    net.load_state_dict(inherited["agent"])
    checkpoint=root/"imitation.pt"
    path=root/"student.json"
    report=json.loads(path.read_text()) if path.exists() else {}
    if "selected_epoch" not in report:
        report=dict(examples={k:len(v) for k,v in data.items()},history=[],initial_validation=measure(net,data["validation"],args.device))
        frequencies = np.bincount(data["train"]["obs"][:,4],minlength=5).astype(float)
        act_weights = np.ones(5)
        if args.balance_acts:
            act_weights[1:] = np.clip(np.sqrt(frequencies[1:].sum()/(4*np.maximum(1,frequencies[1:]))),.55,3.)
        weights=torch.as_tensor(act_weights,dtype=torch.float32,device=args.device)
        report.update(visible_intents=visible_intents,act_weights=act_weights.tolist())
        opt=torch.optim.AdamW(net.parameters(),lr=.00022,weight_decay=.012)
        best=float("inf")
        for epoch in range(1,9):
            net.train()
            for obs,mask,labels in batches(data["train"],rng.permutation(len(data["train"])),args.device):
                logp=net.dist(obs,mask).logits
                weight=weights[obs[:,4].long()]
                loss=-(weight*logp.masked_fill(~labels,-1e9).logsumexp(1)).sum()/weight.sum()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
                opt.step()
            score=measure(net,data["validation"],args.device)
            report["history"].append(dict(epoch=epoch,validation=score))
            criterion = sum(score[a]["nll"]*score[a]["count"]*act_weights[a] for a in range(1,5))/sum(score[a]["count"]*act_weights[a] for a in range(1,5))
            if criterion<best:
                best=criterion
                selected=epoch
                torch.save(net.state_dict(),checkpoint)
            write(root/"training-progress.json",report)
            print("epoch",epoch,score[0],flush=True)
        report["selected_epoch"]=selected
        report["sha256"]=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        write(path,report)
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=report["sha256"]:
        raise ValueError("evaluated checkpoint changed")
    net.load_state_dict(torch.load(checkpoint,map_location=args.device,weights_only=True))
    net.eval()
    if "unseen_demonstrations" not in report:
        report["unseen_demonstrations"]=measure(net,data["test"],args.device)
        write(path,report)
    parent=read_model("build/stratified-gate/challenger.json.gz")
    actor=Actor(net,args.device,apply_model("first",parent),args.threads,visible_intents)
    screen=report.setdefault("screen",{})
    def run(start,count,mode):
        return evaluate(actor,lambda o,m:m,args.device,mode,parent,start,count,192,args.threads,"ironclad")
    for name,blend in (("parent",0.),("standalone",0.),("mixed",.65),("conservative",1.5)):
        if name not in screen:
            actor.blend=blend
            screen[name]=run(224000000,1024,"parent" if name=="parent" else "combat")
            write(path,report)
            print("ordinary screen",name,screen[name],flush=True)
    def rank(row):
        return (len(row["heart_encounter_seeds"]),row["act4"],row["act3"],row["mean_floor"])
    selected=max(("standalone","mixed","conservative"),key=lambda k:rank(screen[k]))
    if rank(screen[selected])<=rank(screen["parent"]):
        report["decision"]="No candidate improved the ordinary-start screen"
        write(path,report)
        return
    actor.blend=dict(standalone=0.,mixed=.65,conservative=1.5)[selected]
    report["selected_play"]=selected
    panels=report.setdefault("test_223M",[])
    for offset in range(sum(v["runs"] for v in panels),32768,2048):
        panels.append(run(223000000+offset,2048,"combat"))
        write(path,report)


if __name__=="__main__":
    main()
