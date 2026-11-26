"""Identify which acts account for the measured gain from the learned map policy."""
import json
from pathlib import Path

import heart_native as engine
import numpy as np

from .heart_curriculum import read_model
from .replay import apply_model
from .train import write


def main():
    out=Path("build/map-act-ablation");out.mkdir(parents=True,exist_ok=True)
    p=out/"report.json";r=json.loads(p.read_text()) if p.exists() else {}
    base=read_model("build/stratified-gate/challenger.json.gz")
    m=read_model("build/route-irl/selected-whole_map.json.gz")
    controls=json.loads(Path("build/route-search-transfer/replicate/report.json").read_text())
    start=controls["start"];ids=np.arange(start,start+32768)
    for name,acts in (("act1",1),("act2",2),("act3",4),("late",6)):
        if name in r:continue
        cfg=[*m["map_policy"][:6],acts];w=apply_model("first",{**base,"map_policy":cfg})
        a=np.concatenate([np.asarray(engine.batch_seeds(w,x.tolist(),4)) for x in np.array_split(ids,16)])
        np.save(out/f"{name}.npy",a)
        hits=set(ids[a[:,6]>0].tolist());ref=set(controls["parent"]["heart_encounter_seeds"])
        r[name]=dict(n=len(a),act2=int((a[:,1]>=2).sum()),act3=int((a[:,1]>=3).sum()),first_boss=int((a[:,0]>=50).sum()),
                     second_boss=int((a[:,0]>=51).sum()),act4=int((a[:,1]>=4).sum()),wins=int(a[:,4].sum()),
                     heart_encounter_seeds=sorted(hits),gained=len(hits-ref),lost=len(ref-hits),truncated=int(a[:,5].sum()))
        write(p,r);print(name,r[name],flush=True)
    r["start"]=start;r["parent"]=controls["parent"];r["all_acts"]=controls["map"];write(p,r)


if __name__=="__main__":
    main()
