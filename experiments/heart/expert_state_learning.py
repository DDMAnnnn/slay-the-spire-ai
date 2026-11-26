"""Learn corrections on the student's own decisions from matched expert combat starts."""
import argparse
import base64
import json
from pathlib import Path
import subprocess

import numpy as np
import sts_env as sts
import torch

from .public_demo_student import batches
from .shared_combat import demo_rows
from .train import write
from .transfer_probe import EXAMPLE, model_code


def send(stream,value):
    stream.write(json.dumps(value,separators=(",",":"))+"\n");stream.flush()


def play(worker,net,device,split,collect=None,search_budget=None,search_mode=-1,afterstate=None):
    request=dict(split=split,labels=collect is not None)
    if search_budget is not None:request.update(search_budget=search_budget,search_mode=search_mode)
    if afterstate is not None:request["afterstate"]=afterstate
    send(worker.stdin,request)
    dtype=np.dtype([("obs","<i2",sts.OBS_SIZE),("legal","u1",81),("action","u1")])
    net.eval()
    with torch.inference_mode():
        while True:
            line=worker.stdout.readline()
            if not line:raise RuntimeError(f"source simulator closed the pipe: {worker.poll()}")
            msg=json.loads(line)
            if "fights" in msg:return msg["fights"]
            data=np.frombuffer(base64.b64decode(msg["data"]),dtype=dtype)
            if len(data)!=msg["rows"]:raise ValueError("corrupt observation batch")
            obs=torch.as_tensor(data["obs"].astype(np.float32),device=device)
            mask=torch.zeros(len(obs),sts.N_ACTIONS,dtype=torch.bool,device=device)
            mask[:,:81]=torch.as_tensor(data["legal"].astype(bool),device=device)
            actions=net.dist(obs,mask).logits.argmax(1).tolist()
            if collect is not None:collect.extend(data[data["action"]!=255].copy())
            send(worker.stdin,actions)


def statistics(fights):
    groups={}
    for act in range(5):
        items=[f for f in fights if not act or f["act"]==act]
        if not items:continue
        groups[str(act) if act else "all"]=dict(fights=len(items),wins=sum(f["win"] for f in items),
            expert_wins=sum(f["ref_win"] for f in items),exit_hp=sum(f["hp"] for f in items)/len(items),
            expert_exit_hp=sum(f["ref_hp"] for f in items)/len(items), capped=sum(f["capped"] for f in items),
            followups=sum(f["followups"] for f in items),queries=sum(f["labels"] for f in items))
    return groups


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tiny",action="store_true")
    p.add_argument("--out",type=Path,default=Path("build/expert-state-learning"))
    args=p.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    file=out/"report.json";r=json.loads(file.read_text()) if file.exists() else {}
    torch.manual_seed(239044);torch.set_num_threads(2)
    device="cuda" if torch.cuda.is_available() else "cpu"
    source=Path("build/public-expert-intent")
    offline=demo_rows(source,"train")
    base=source/"imitation.pt"
    net,_=model_code().load(EXAMPLE/"weights.pt",device)
    def restore(checkpoint):net.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True))
    restore(base)
    cmd=[str(Path("build/public-expert-traces/python313/python.exe").resolve()),str(Path(__file__).with_name("expert_state_worker.py"))]
    if args.tiny:cmd.append("--tiny")
    with subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,encoding="ascii",bufsize=1) as worker:
        try:
            r.update(json.loads(worker.stdout.readline()));write(file,r)
            def score(name,split):
                bucket=r.setdefault(split,{})
                if name not in bucket:
                    rows=play(worker,net,device,split)
                    write(out/f"{split}-{name}.json",rows)
                    bucket[name]=statistics(rows);write(file,r)
                    print(split,name,bucket[name]["all"],flush=True)
                return bucket[name]
            score("parent","validation")
            f=out/"corrections.npy"
            if not f.exists():
                corrections=[];rows=play(worker,net,device,"train",corrections)
                np.save(f,np.array(corrections));r["collection"]=statistics(rows);write(file,r)
            corrections=np.load(f)
            if not len(corrections):raise ValueError("no corrections collected")
            o_act=offline["obs"][:,4]
            offline_groups={a:np.flatnonzero(o_act==a) for a in range(1,5)}
            # Both students see equal counts of each act and use identical optimization schedules.
            for mode in ("offline","correction"):
                rng=np.random.default_rng(23937);restore(base)
                opt=torch.optim.AdamW(net.parameters(),lr=.000075,weight_decay=.012)
                for epoch in range(1,7):
                    net.train()
                    for ix in np.array_split(rng.permutation(len(corrections)),max(1,int(np.ceil(len(corrections)/96)))):
                        sel=corrections[ix]
                        ids=np.array([rng.choice(offline_groups[int(a)]) for a in sel["obs"][:,4]])
                        control=np.array([rng.choice(offline_groups[int(a)]) for a in sel["obs"][:,4]])
                        chunks=[next(batches(offline,ids,device)),next(batches(offline,control,device)) if mode=="offline"
                                else next(batches(sel,np.arange(len(sel)),device))]
                        obs,mask,labels=[torch.cat(pair) for pair in zip(*chunks)]
                        logp=net.dist(obs,mask).logits;loss=-logp.masked_fill(~labels,-1e9).logsumexp(1).mean()
                        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
                    if epoch in (1,3,6):
                        name=f"{mode}-{epoch}";torch.save(net.state_dict(),out/f"{name}.pt")
                        score(name,"validation")
            for mode in ("offline","correction"):
                names=[k for k in r["validation"] if k.startswith(mode)]
                def rank(k):
                    s=r["validation"][k]
                    return (sum(s[str(a)]["wins"]/s[str(a)]["fights"] for a in range(1,5)),s["all"]["exit_hp"])
                best=max(names,key=rank);r.setdefault("chosen",{})[mode]=best
            for name in ["parent",*r["chosen"].values()]:
                restore(base if name=="parent" else out/f"{name}.pt");score(name,"test")
            write(file,r)
        finally:
            if worker.poll() is None:
                try:send(worker.stdin,{"stop":True})
                except (BrokenPipeError,OSError):pass
                worker.stdin.close()


if __name__=="__main__":
    main()
