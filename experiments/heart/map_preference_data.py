"""Collect action-conditioned run outcomes for route improvement, with an unchanged-action replay control."""
import gzip
import json
from pathlib import Path

import heart_native as engine
import numpy as np

from .heart_curriculum import read_model
from .replay import apply_model
from .train import write


def main():
    out=Path("build/map-preferences");out.mkdir(parents=True,exist_ok=True)
    parent=read_model("build/route-irl/selected-whole_map.json.gz");w=apply_model("first",parent)
    start=251000000;ids=np.arange(start,start+32768)
    file=out/"scout.npy"
    if not file.exists():np.save(file,np.concatenate([np.asarray(engine.batch_seeds(w,part.tolist(),3)) for part in np.array_split(ids,32)]))
    scout=np.load(file)
    rng=np.random.default_rng(25117)
    report={"scout_start":start,"scout_runs":len(ids),"acts":{}}
    for act in (1,2,3):
        possible=np.flatnonzero(scout[:,1]>=act)
        selected=np.sort(rng.choice(possible,min(2400,len(possible)),replace=False))
        path=out/f"act-{act}.jsonl.gz"
        if not path.exists():
            tmp=path.with_suffix(".tmp")
            with gzip.open(tmp,"wt") as fp:
                for ix in np.array_split(selected,max(1,len(selected)//128)):
                    for ctx,board,opts in engine.map_preferences(w,ids[ix].tolist(),act,3):
                        if not ctx:continue
                        k=ix[ctx[0]]
                        if any(o[-1] for o in opts) or scout[k,5]:continue
                        control=next(o for o in opts if o[0]==ctx[1])
                        expected=scout[k,[0,1,2,6,4,5]].tolist()
                        if control[1:]!=expected:raise ValueError(f"parent continuation differs on seed {ids[k]}: {control[1:]} {expected}")
                        ctx[0]=int(ids[k]);fp.write(json.dumps([ctx,board,opts],separators=(",",":"))+"\n")
            tmp.replace(path)
        with gzip.open(path,"rt") as src:examples=list(map(json.loads,src))
        report["acts"][str(act)]=dict(forks=len(examples),continuations=sum(len(r[2]) for r in examples),
            heart_disagreements=sum(len({a[4] for a in r[2]})>1 for r in examples),
            act_disagreements=sum(len({a[2] for a in r[2]})>1 for r in examples))
        write(out/"collection.json",report);print(act,report["acts"][str(act)],flush=True)


if __name__=="__main__":
    main()
