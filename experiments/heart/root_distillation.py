"""Compare hard, visit and value-informed supervision on exactly the same public search roots."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
import sts_env as sts

from .expert_state_learning import play,send,statistics
from .matched_combat_rl import inputs
from .shared_combat import demo_rows
from .train import write
from .transfer_probe import EXAMPLE,model_code


def canonical(row):
    # Equal visible copies of a card targeting the same enemy form one action class.
    off,n,w=sts.OBS_LAYOUT["hand"];hand=row["obs"][off:off+n*w].reshape(n,w)
    mapping=np.arange(81,dtype=np.int64);prior={}
    for a in np.flatnonzero(row["legal"][:50]):
        key=(hand[a//5].tobytes(),a%5)
        mapping[a]=prior.setdefault(key,a)
    return mapping


def prepare(root):
    all_targets={}
    for f in sorted(root.glob("roots-*.jsonl.gz")):
        with gzip.open(f,"rt") as src:
            for seed,step,act,best,estimates in map(json.loads,src):
                k=seed,step
                if k in all_targets:raise ValueError("two records for one source decision")
                all_targets[k]=best,estimates
    pools={}
    for split in ("train","validation","test"):
        demo=demo_rows(Path("build/public-expert-intent"),split)
        indexes=[];maps=[];original=[];teacher=[];soft=[];visits=[];acts=Counter()
        for i,r in enumerate(demo):
            target=all_targets.pop((int(r["seed"]),int(r["step"])),None)
            if target is None:continue
            best,estimates=target
            m=canonical(r);cnt=np.zeros(81);reward=np.zeros(81)
            for a,n,q in estimates:
                if not r["legal"][a]:raise ValueError("teacher scored an illegal local action")
                cnt[m[a]]+=n;reward[m[a]]+=n*q
            legal=cnt>0
            if legal.sum()<2 or not legal[m[best]]:continue
            mean=reward.sum()/cnt.sum()
            # Dampen small-sample estimates and standardize within a position.
            value=(reward[legal]+4*mean)/(cnt[legal]+4)
            z=(value-value.mean())/max(1e-6,value.std())
            target=np.zeros(81,np.float32)
            mass=np.exp((z-z.max())/.8);target[legal]=mass/mass.sum()
            v=(cnt/cnt.sum()).astype(np.float32)
            indexes.append(i);maps.append(m.astype(np.uint8));original.append(m[r["action"]]);teacher.append(m[best])
            soft.append(.75*target+.25*v);visits.append(v);acts[int(r["obs"][4])]+=1
        pools[split]=(demo,np.asarray(indexes),np.stack(maps),np.array(original),np.array(teacher),np.stack(soft),np.stack(visits))
        print("roots",split,dict(acts),flush=True)
    if all_targets:raise ValueError(f"{len(all_targets)} root labels had no matching verified public observation")
    return pools


def objective(net,pool,ids,kind,device):
    demo,indices,mapping,orig,best,soft,counts=pool
    rows=demo[indices[ids]]
    obs,legal=inputs(rows,device)
    p=net.dist(obs,legal).probs[:,:81]
    mapping=torch.as_tensor(mapping[ids].astype(np.int64),device=device)
    classes=torch.zeros_like(p).scatter_add(1,mapping,p)
    ar=torch.arange(len(ids),device=device)
    orig=torch.as_tensor(orig[ids],device=device);best=torch.as_tensor(best[ids],device=device)
    safe=classes.clamp_min(1e-10)
    demo_loss=-safe[ar,orig].log()
    if kind=="hard":
        label_loss=-safe[ar,best].log()
    else:
        y=torch.as_tensor((soft if kind=="value" else counts)[ids],device=device)
        # Absence from the search tree is missing evidence. Evaluate relative
        # student probabilities on compared alternatives only.
        compared=classes*(y>0)
        logp=(compared/compared.sum(1,keepdim=True).clamp_min(1e-10)).clamp_min(1e-10).log()
        label_loss=-(y*logp).sum(1)
    return (.35*demo_loss+.65*label_loss).mean(),demo_loss.sum(),(classes.argmax(1)==orig).sum()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,default=Path("build/public-root-compatible"))
    args=p.parse_args();root=args.root
    if len(list(root.glob("roots-*.jsonl.gz")))!=4:raise ValueError("four completed and verified shards required")
    pools=prepare(root)
    device="cuda" if torch.cuda.is_available() else "cpu";torch.set_num_threads(2)
    net,_=model_code().load(EXAMPLE/"weights.pt",device)
    source=Path("build/public-expert-intent/imitation.pt")
    def restore(p):net.load_state_dict(torch.load(p,map_location=device,weights_only=True))
    file=root/"report.json";r=json.loads(file.read_text()) if file.exists() else {"fits":{},"validation":{},"test":{}}
    r["roots"]={k:len(v[1]) for k,v in pools.items()}
    for kind in ("hard","visits","value"):
        if kind in r["fits"]:continue
        torch.manual_seed(247149);rng=np.random.default_rng(247149);restore(source)
        opt=torch.optim.AdamW(net.parameters(),lr=.000085,weight_decay=.012)
        history=[];best=float("inf")
        for ep in range(1,7):
            net.train()
            ids=rng.permutation(len(pools["train"][1]))
            for ix in np.array_split(ids,max(1,int(np.ceil(len(ids)/192)))):
                loss,_,_=objective(net,pools["train"],ix,kind,device)
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
            net.eval();a=b=0.
            with torch.no_grad():
                for ix in np.array_split(np.arange(len(pools["validation"][1])),12):
                    _,n,correct=objective(net,pools["validation"],ix,kind,device);a+=n.item();b+=correct.item()
            n=len(pools["validation"][1]);entry=dict(epoch=ep,demo_nll=a/n,demo_accuracy=b/n);history.append(entry)
            if a<best:best=a;chosen=ep;torch.save(net.state_dict(),root/f"{kind}.pt")
        r["fits"][kind]=dict(chosen=chosen,history=history);write(file,r)
    cmd=[str(Path("build/public-expert-traces/python313/python.exe").resolve()),str(Path(__file__).with_name("expert_state_worker.py"))]
    with subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,encoding="ascii",bufsize=1) as worker:
        try:
            r.update(json.loads(worker.stdout.readline()))
            for split in ("validation","test"):
                for name in ("parent","hard","visits","value"):
                    if name in r[split]:continue
                    cache=root/f"{split}-{name}.json"
                    if cache.exists():
                        r[split][name]=statistics(json.loads(cache.read_text()))
                        write(file,r)
                        continue
                    restore(source if name=="parent" else root/f"{name}.pt")
                    fights=play(worker,net,device,split)
                    r[split][name]=statistics(fights);write(root/f"{split}-{name}.json",fights);write(file,r)
                    print(split,name,r[split][name]["all"],flush=True)
        finally:
            if worker.poll() is None:
                try:send(worker.stdin,{"stop":True})
                except (BrokenPipeError,OSError):pass
                worker.stdin.close()


if __name__=="__main__":
    main()
