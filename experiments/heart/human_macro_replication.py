"""Replicate independently selected human event and campfire changes on the same ordinary starts."""
import json
from pathlib import Path

import heart_native as engine
import numpy as np

from .heart_curriculum import read_model, save_model
from .replay import apply_model
from .route_rollouts import summarize
from .train import write


def main():
    out=Path("build/human-macro-replication");out.mkdir(exist_ok=True,parents=True)
    parent=read_model("build/route-irl/selected-whole_map.json.gz")
    events=read_model("build/human-events/challenger.json.gz")["human_events"]
    fire=read_model("build/human-campfires/selected-light.json.gz")["resource_policy"]
    groups={"parent":parent,"combined":{**parent,"human_events":events,"resource_policy":fire},
            "event":{**parent,"human_events":events},"fire":{**parent,"resource_policy":fire}}
    path=out/"report.json"
    r=json.loads(path.read_text()) if path.exists() else {}
    seed=260117;ids=np.random.default_rng(seed).bit_generator.random_raw(32768)
    for name,m in groups.items():
        if name in r:continue
        w=apply_model("first",m);blocks=[]
        for off in range(0,len(ids),1024):
            part=out/f"{name}-{off}.npy"
            if not part.exists():np.save(part,np.asarray(engine.batch_seeds(w,ids[off:off+1024].tolist(),3)))
            blocks.append(np.load(part))
        rows=np.concatenate(blocks)
        hits=set(map(str,ids[rows[:,6]>0]))
        r[name]=dict(**summarize(rows),heart_encounter_seeds=sorted(hits),victory_seeds=list(map(str,ids[rows[:,4]>0])))
        if name!="parent":
            old=set(r["parent"]["heart_encounter_seeds"])
            r[name].update(gained=len(hits-old),lost=len(old-hits))
        save_model(out/f"{name}.json.gz",m);write(path,r);print(name,summarize(rows),flush=True)


if __name__=="__main__":
    main()
