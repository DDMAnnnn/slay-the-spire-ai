"""Test human campfire and smith policies on matched, uniformly sampled 64-bit starts."""
import json
from pathlib import Path

import heart_native as engine
import numpy as np

from .heart_curriculum import read_model, save_model
from .replay import apply_model
from .route_rollouts import summarize
from .train import write


def main():
    out=Path("build/human-campfires")
    base=read_model("build/route-irl/selected-whole_map.json.gz")
    human=read_model(out/"human.json.gz")["resource_policy"][:-1]
    path=out/"play.json"
    r=json.loads(path.read_text()) if path.exists() else {"selection":{},"test":{}}
    def run(phase, tag, mix, n):
        if tag in r[phase]:return
        seed=256021 if phase=="selection" else 257021
        seeds=np.random.default_rng(seed).bit_generator.random_raw(n)
        if len(np.unique(seeds))!=n:raise ValueError("duplicate game seed")
        cfg={**base,"resource_policy":[*human,[mix[0],0.,mix[1],0.]]}
        w=apply_model("first",cfg)
        parts=[]
        for i in range(0,n,1024):
            p=out/f"{phase}-{tag}-{i}.npy"
            if not p.is_file():
                np.save(p,np.asarray(engine.batch_seeds(w,seeds[i:i+1024].tolist(),3)))
            parts.append(np.load(p))
        rows=np.concatenate(parts)
        r[phase][tag]=dict(mix=list(mix),**summarize(rows),heart_encounter_seeds=[str(s) for s in seeds[rows[:,6]>0]],
                           victory_seeds=[str(s) for s in seeds[rows[:,4]>0]])
        if phase=="test" and tag!="parent":
            a=set(r[phase][tag]["heart_encounter_seeds"]);b=set(r[phase]["parent"]["heart_encounter_seeds"])
            r[phase][tag].update(gained=len(a-b),lost=len(b-a))
            save_model(out/f"selected-{tag}.json.gz",cfg)
        write(path,r);print(phase,tag,summarize(rows),flush=True)
    candidates={"light":(.35,.35),"strong":(.75,.75),"full":(1.,1.)}
    run("selection","parent",(0,0),8192)
    for k,v in candidates.items():run("selection",k,v,8192)
    def rank(k):
        v=r["selection"][k]
        return v["encounters"],v["act4"],v["second"],v["boss"]
    winner=max(candidates,key=rank);r["selected"]=winner
    run("test","parent",(0,0),32768)
    run("test",winner,candidates[winner],32768)
    candidate=r["test"][winner]["encounters"];original=r["test"]["parent"]["encounters"]
    if candidate>=max(original+4,1.25*original):
        run("test","camp_only",(candidates[winner][0],0),32768)
        run("test","smith_only",(0,candidates[winner][1]),32768)
    write(path,r)


if __name__=="__main__":
    main()
