"""Evaluate a combat specialist only after its shared, deterministic early-game prefix."""
import argparse
import hashlib
import json
from pathlib import Path

import heart_native as engine
import numpy as np
import sts_env as sts
import torch

from .heart_curriculum import read_model
from .public_demo_student import Actor
from .replay import apply_model
from .train import write
from .transfer_probe import EXAMPLE, model_code


@torch.inference_mode()
def play(seeds, floor, actor, device, threads, max_act=4):
    env = sts.VecEnv(len(seeds),20,threads,3500,True,False,"ironclad",0.,0,0.)
    env.reset(seeds.tolist())
    active = np.ones(len(seeds),bool)
    reached = np.zeros(len(seeds),bool)
    result = np.zeros((len(seeds),5),int)
    off,n,width = sts.OBS_LAYOUT["monsters"]
    heart = sts.MONSTER_NAMES.index("CORRUPT_HEART")+1
    while active.any():
        obs,mask=np.asarray(env.obs),np.asarray(env.mask)
        if not np.array_equal(np.asarray(env.seed)[active],seeds[active]):
            raise RuntimeError("a seed changed before settlement")
        monsters=obs[:,off:off+n*width].reshape(len(seeds),n,width)
        reached |= active & (obs[:,4]==4) & (obs[:,7]>0) & ((monsters[:,:,0]==heart)&(monsters[:,:,6]>0)).any(1)
        free = active & (mask.sum(1)>1)
        use = free & (obs[:,3]>=floor) & (obs[:,4]<=max_act) & (obs[:,7]>0) & (mask[:,80]>0)
        act = mask.argmax(1).astype(np.int32)
        if (free&~use).any():
            ix=free&~use
            act[ix]=engine.teacher_batch(actor.weights,obs[ix],mask[ix],threads)
        if use.any():
            o=torch.as_tensor(obs[use],device=device)
            m=torch.as_tensor(mask[use],dtype=torch.bool,device=device)
            act[use]=actor.dist(o,m).logits.argmax(1).cpu().numpy()
        if env.step(act):
            raise RuntimeError("illegal decision")
        finished=active & (np.asarray(env.terminated,dtype=bool)|np.asarray(env.truncated,dtype=bool))
        for i in np.flatnonzero(finished):
            last=env.final_obs[i]
            result[i]=[last[3],last[4],reached[i],env.won[i],env.truncated[i]]
        active[finished]=False
    return result


def summary(rows):
    return dict(n=len(rows),hits=int(rows[:,2].sum()),wins=int(rows[:,3].sum()),act3=int((rows[:,1]>=3).sum()),
                act4=int((rows[:,1]>=4).sum()),second_boss=int((rows[:,0]>=51).sum()),truncated=int(rows[:,4].sum()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=Path("build/public-expert-traces"))
    parser.add_argument("--out",type=Path,default=Path("build/late-demo"))
    parser.add_argument("--threads",type=int,default=2)
    parser.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    args=parser.parse_args()
    torch.set_num_threads(2)
    out=args.out;out.mkdir(exist_ok=True)
    src=args.source
    path=out/"report.json"
    r=json.loads(path.read_text()) if path.exists() else {"selection":{},"test":{}}
    student=json.loads((src/"student.json").read_text())
    if hashlib.sha256((src/"imitation.pt").read_bytes()).hexdigest()!=student["sha256"]:
        raise ValueError("unfrozen teacher-imitation checkpoint")
    macro=read_model("build/stratified-gate/challenger.json.gz")
    w=apply_model("first",macro)
    net,_=model_code().load(EXAMPLE/"weights.pt",args.device)
    net.load_state_dict(torch.load(src/"imitation.pt",weights_only=True,map_location=args.device))
    actor=Actor(net.eval(),args.device,w,args.threads,student.get("visible_intents",False))
    start=225000000
    f=out/"selection-parent.npy"
    if not f.exists():
        np.save(f,np.asarray(engine.batch_seeds(w,list(range(start,start+32768)),args.threads)))
    ref=np.load(f)
    for cutoff in (35,49):
        possible=np.flatnonzero(ref[:,0]>=cutoff)
        ids=np.sort(np.random.default_rng(225901+cutoff).choice(possible,min(len(possible),192),replace=False))
        x=ref[ids]
        baseline=x[:,[0,1,6,4,5]]
        group=r["selection"].setdefault(str(cutoff),dict(eligible=len(possible),original=summary(baseline),arms={}))
        if "same_prefix_control" not in group:
            check=play(start+ids[:24],99,actor,args.device,args.threads)
            if not np.array_equal(check,baseline[:24]):
                raise RuntimeError("vector replay and native parent outcomes differ")
            group["same_prefix_control"]=len(check)
        for name,blend in (("standalone",0.),("mixed",.65),("conservative",1.5)):
            if name in group["arms"]:
                continue
            actor.blend=blend
            rows=play(start+ids,cutoff,actor,args.device,args.threads)
            gain=summary(rows)
            weight=len(possible)/len(ids)
            gain["estimated_extra_entries"]=weight*(gain["hits"]-group["original"]["hits"])
            gain["estimated_extra_act4"]=weight*(gain["act4"]-group["original"]["act4"])
            gain["estimated_extra_second_boss"]=weight*(gain["second_boss"]-group["original"]["second_boss"])
            group["arms"][name]=gain
            write(path,r)
            print("selection",cutoff,name,gain,flush=True)
    pairs=[(int(k),a,v) for k,g in r["selection"].items() for a,v in g["arms"].items()]
    cutoff,name,score=max(pairs,key=lambda p:tuple(p[2][k] for k in
        ("estimated_extra_entries","estimated_extra_act4","estimated_extra_second_boss")))
    r["chosen"]={"floor":cutoff,"blend":name}
    # The final panel is untouched by the specialist's training, checkpoint and floor selection.
    start=226000000
    f=out/"test-parent.npy"
    if not f.exists():
        np.save(f,np.asarray(engine.batch_seeds(w,list(range(start,start+32768)),args.threads)))
    ref=np.load(f)
    ids=np.flatnonzero(ref[:,0]>=cutoff)
    actor.blend=dict(standalone=0.,mixed=.65,conservative=1.5)[name]
    r["test_parent"]=summary(ref[:,[0,1,6,4,5]])
    r["eligible"]=len(ids)
    parts=[]
    for i in range(0,len(ids),128):
        chunk=ids[i:i+128]
        f=out/f"test-{cutoff}-{name}-{i}.npy"
        if not f.exists():
            np.save(f,play(start+chunk,cutoff,actor,args.device,args.threads))
        parts.append(np.load(f))
    results=ref[:,[0,1,6,4,5]].copy()
    if results[ref[:,0]<cutoff,2].any():
        raise RuntimeError("an excluded run could still have reached the Heart")
    results[ids]=np.concatenate(parts)
    r["test"]=summary(results)
    r["start"]=start
    r["heart_encounter_seeds"]=(start+np.flatnonzero(results[:,2])).tolist()
    r["parent_encounter_seeds"]=(start+np.flatnonzero(ref[:,6])).tolist()
    r["victory_seeds"]=(start+np.flatnonzero(results[:,3])).tolist()
    write(path,r)
    print("ordinary panel",r["test_parent"],r["test"],flush=True)


if __name__=="__main__":
    main()
