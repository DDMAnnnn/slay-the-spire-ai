"""Measure zero-shot Defect-to-Ironclad transfer on a fixed range of ordinary starts."""
import argparse
import hashlib
import importlib.util
from pathlib import Path
import sys
from time import perf_counter

import heart_native as native
import numpy as np
import sts_env as sts
import torch

from .heart_curriculum import read_model
from .recovery import panel
from .replay import apply_model
from .train import write

TOOLKIT=Path("build/rl-toolkit")
EXAMPLE=TOOLKIT/"examples/defect_a20"


def model_code():
    sys.path.insert(0,str((TOOLKIT/"bindings").resolve()))
    spec=importlib.util.spec_from_file_location("published_defect_agent",EXAMPLE/"agent.py")
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@torch.inference_mode()
def evaluate(actor,unpick,device,mode,parent,start,count,batch,threads,character):
    n=min(count,batch)
    env=sts.VecEnv(n,20,threads,3500,True,False,character,0.,0,0.)
    env.reset(list(range(start,start+n)))
    flags=np.zeros(n,dtype=bool)
    records={}
    monster=sts.MONSTER_NAMES.index("CORRUPT_HEART")+1
    offset,height,width=sts.OBS_LAYOUT["monsters"]
    weights=apply_model("first",parent)
    total=0
    t=perf_counter()
    while len(records)<count:
        seeds=np.asarray(env.seed).copy()
        active=seeds<start+count
        obs=np.asarray(env.obs)
        mask=np.asarray(env.mask)
        mons=obs[:,offset:offset+height*width].reshape(n,height,width)
        flags|=active & (obs[:,4]==4) & (obs[:,7]>0) & ((mons[:,:,0]==monster)&(mons[:,:,6]>0)).any(1)
        # Native slots re-seed by stride. Finish every prescribed seed, including long runs.
        # Extra seeds only occupy drained slots and contribute no observation or result.
        acts=mask.argmax(1).astype(np.int32)
        choices=active & (mask.sum(1)>1)
        use=choices.copy()
        if mode=="combat":
            use &= obs[:,7]>0
        elif mode=="macro":
            use &= obs[:,7]==0
        elif mode=="parent":
            use[:]=False
        prior=choices & ~use
        if prior.any():
            acts[prior]=native.teacher_batch(weights,obs[prior],mask[prior],threads)
        if use.any():
            o=torch.as_tensor(obs[use],device=device)
            m=unpick(o,torch.as_tensor(mask[use],dtype=torch.bool,device=device))
            acts[use]=actor.dist(o,m).logits.argmax(1).cpu().numpy()
        if env.step(acts):
            raise RuntimeError("invalid action in transfer test")
        done=np.asarray(env.terminated,dtype=bool)|np.asarray(env.truncated,dtype=bool)
        for i in np.flatnonzero(active & done):
            end=env.final_obs[i]
            records[int(seeds[i])]=[int(end[3]),int(end[4]),int(flags[i]),int(env.won[i]),int(env.truncated[i])]
        flags[done]=False
        total+=int(active.sum())
    assert sorted(records)==list(range(start,start+count))
    r=np.asarray([records[s] for s in sorted(records)])
    return dict(runs=count,seconds=round(perf_counter()-t,2),active_steps=total,
                mean_floor=float(r[:,0].mean()),act2=int((r[:,1]>=2).sum()),act3=int((r[:,1]>=3).sum()),
                act4=int((r[:,1]>=4).sum()),heart_encounter_seeds=(start+np.flatnonzero(r[:,2])).tolist(),
                victory_seeds=(start+np.flatnonzero(r[:,3])).tolist(),truncated=int(r[:,4].sum()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=Path("build/transfer-probe/report.json"))
    p.add_argument("--start",type=int,default=155000000)
    p.add_argument("--count",type=int,default=512)
    p.add_argument("--batch",type=int,default=192)
    p.add_argument("--threads",type=int,default=6)
    p.add_argument("--mode",nargs="+",choices=("all","combat","macro"),default=["all"])
    p.add_argument("--character",choices=("ironclad","defect"),default="ironclad")
    p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    args=p.parse_args()
    if min(args.count,args.threads,args.batch)<1: p.error("positive dimensions required")
    torch.set_num_threads(3)
    torch.distributions.Distribution.set_default_validate_args(False)
    source=model_code()
    weights=EXAMPLE/"weights.pt"
    actor,cfg=source.load(weights,args.device)
    parent=read_model("build/trajectory-teacher/challenger.json.gz")
    report=dict(start=args.start,runs=args.count,character=args.character,device=args.device,
                action_selection="greedy",weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
                source_character=cfg["character"],source_training_steps=cfg["steps_trained"],results={})
    args.out.parent.mkdir(parents=True,exist_ok=True)
    if args.character=="ironclad":
        report["parent"],_=panel(parent,args.start,args.count,args.threads)
    for mode in args.mode:
        r=evaluate(actor,source.no_unpick,args.device,mode,parent,args.start,args.count,args.batch,args.threads,args.character)
        report["results"][mode]=r
        print(mode,r,flush=True)
        write(args.out,report)


if __name__=="__main__":
    main()
