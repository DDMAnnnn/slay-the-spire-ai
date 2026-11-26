"""Optimize combat survival by comparing sampled plays from identical expert-reached states."""
import argparse
import base64
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
import sts_env as sts

from .expert_state_learning import play, send, statistics
from .train import write
from .transfer_probe import EXAMPLE, model_code


def inputs(rows,device):
    obs=torch.as_tensor(rows["obs"].astype(np.float32),device=device)
    mask=torch.zeros(len(rows),sts.N_ACTIONS,device=device,dtype=torch.bool)
    mask[:,:81]=torch.as_tensor(rows["legal"].astype(bool),device=device)
    return obs,mask


def rollout(worker,model,device,iteration):
    send(worker.stdin,dict(split="train",subset=80,subset_seed=241901+iteration,repeats=4,episodes=True))
    dtype=np.dtype([("obs","<i2",sts.OBS_SIZE),("legal","u1",81),("label","u1")])
    chunks=[];actions=[];old_logp=[];episodes=[]
    model.eval()
    with torch.inference_mode():
        while True:
            msg=json.loads(worker.stdout.readline())
            if "fights" in msg:break
            rows=np.frombuffer(base64.b64decode(msg["data"]),dtype=dtype)
            dist=model.dist(*inputs(rows,device))
            a=dist.sample()
            chunks.append(rows);actions.append(a.cpu().numpy())
            old_logp.append(dist.log_prob(a).cpu().numpy());episodes.extend(msg["episodes"])
            send(worker.stdin,a.tolist())
    f=msg["fights"]
    if len(f)%4:raise ValueError("partial group of cloned starts")
    for i in range(0,len(f),4):
        if len({(v["seed"],v["index"]) for v in f[i:i+4]})!=1:raise ValueError("initial states differ within group")
    # A fight win dominates any difference in remaining health. No future information enters an action.
    returns=np.array([float(v["win"])+.25*min(1.,v["hp"]/max(1,v["max_hp"])) for v in f],dtype=np.float32).reshape(-1,4)
    advantage=(returns-returns.mean(1,keepdims=True))*4/3
    info=dict(fights=len(f),wins=sum(v["win"] for v in f),informative_groups=int((np.ptp(returns,axis=1)>0).sum()),
              groups=len(returns),decisions=len(episodes),mean_return=float(returns.mean()),capped=sum(v["capped"] for v in f))
    return np.concatenate(chunks),np.concatenate(actions),np.concatenate(old_logp),advantage.ravel()[episodes],info


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tiny",action="store_true");p.add_argument("--out",type=Path,default=Path("build/matched-combat-rl"))
    args=p.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    path=out/"report.json"
    r=json.loads(path.read_text()) if path.exists() else {"updates":[],"validation":{},"test":{}}
    torch.manual_seed(24191);torch.set_num_threads(2);rng=np.random.default_rng(24191)
    device="cuda" if torch.cuda.is_available() else "cpu"
    model,_=model_code().load(EXAMPLE/"weights.pt",device)
    anchor,_=model_code().load(EXAMPLE/"weights.pt",device)
    base=Path("build/public-expert-intent/imitation.pt")
    def restore(net,p):net.load_state_dict(torch.load(p,map_location=device,weights_only=True))
    restore(anchor,base);anchor.eval()
    restore(model,base)
    opt=torch.optim.AdamW(model.parameters(),lr=.000045,weight_decay=.012)
    cmd=[str(Path("build/public-expert-traces/python313/python.exe").resolve()),str(Path(__file__).with_name("expert_state_worker.py"))]
    if args.tiny:cmd.append("--tiny")
    limit=2 if args.tiny else 12
    with subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,encoding="ascii",bufsize=1) as worker:
        try:
            r.update(json.loads(worker.stdout.readline()))
            def assess(split,name):
                if name in r[split]:return
                fights=play(worker,model,device,split)
                r[split][name]=statistics(fights);write(out/f"{split}-{name}.json",fights);write(path,r)
                print(split,name,r[split][name]["all"],flush=True)
            assess("validation","parent")
            for it in range(1,limit+1):
                name=f"update-{it}";state=out/f"{name}.pt"
                if it<=len(r["updates"]):
                    saved=torch.load(state,map_location=device,weights_only=True)
                    model.load_state_dict(saved["model"]);opt.load_state_dict(saved["optimizer"])
                else:
                    torch.manual_seed(24191+it);rng=np.random.default_rng(24191+it)
                    data,action,logp,adv,info=rollout(worker,model,device,it)
                    model.eval()  # Stochastic actions supply exploration; dropout would change the behavior likelihood.
                    if info["informative_groups"]:
                        adv=adv/max(float(adv.std()),.05)
                        for _ in range(2):
                            for ix in np.array_split(rng.permutation(len(data)),max(1,int(np.ceil(len(data)/192)))):
                                obs,mask=inputs(data[ix],device)
                                a=torch.as_tensor(action[ix],device=device);v=torch.as_tensor(adv[ix],device=device)
                                dist=model.dist(obs,mask)
                                ratio=(dist.log_prob(a)-torch.as_tensor(logp[ix],device=device)).exp()
                                with torch.no_grad():ref=anchor.dist(obs,mask)
                                kl=torch.distributions.kl_divergence(dist,ref).mean()
                                loss=-torch.minimum(ratio*v,ratio.clamp(.85,1.15)*v).mean()+.08*kl
                                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),.8);opt.step()
                    torch.save({"model":model.state_dict(),"optimizer":opt.state_dict()},state)
                    r["updates"].append(info);write(path,r);print(name,info,flush=True)
                if it in (1,3,6,9,12) or it==limit:assess("validation",name)
            def rank(name):
                g=r["validation"][name]
                return sum(g[str(a)]["wins"]/g[str(a)]["fights"] for a in range(1,5)),g["all"]["exit_hp"]
            chosen=max((k for k in r["validation"] if k!="parent"),key=rank)
            restore(model,base);assess("test","parent")
            model.load_state_dict(torch.load(out/f"{chosen}.pt",map_location=device,weights_only=True)["model"])
            assess("test",chosen);r["chosen"]=chosen
            torch.save(model.state_dict(),out/"selected.pt");write(path,r)
        finally:
            if worker.poll() is None:
                try:send(worker.stdin,{"stop":True})
                except (BrokenPipeError,OSError):pass
                worker.stdin.close()


if __name__=="__main__":
    main()
