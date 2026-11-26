"""Test learned event decisions together with human campfire choices, then isolate successful parts."""
import json
from pathlib import Path

import heart_native as engine
import numpy as np

from .heart_curriculum import read_model, save_model
from .replay import apply_model
from .route_rollouts import summarize
from .train import write


def main():
    out=Path("build/human-events")
    path=out/"play.json"
    r=json.loads(path.read_text()) if path.exists() else {"selection":{},"test":{}}
    base=read_model("build/route-irl/selected-whole_map.json.gz")
    fire=read_model("build/human-campfires/human.json.gz")["resource_policy"][:-1]
    variants={"parent":base}
    for name in ("frequency","context"):
        weights=read_model(out/f"{name}.json.gz")["human_events"][:-1]
        for blend in (.45,.85):
            tag=f"{name}-{blend}"
            variants[tag]={**base,"human_events":[*weights,blend]}
            variants[tag+"-camp"]={**variants[tag],"resource_policy":[*fire,[.75,0.,.75,0.]]}
    variants["camp"]={**base,"resource_policy":[*fire,[.75,0.,.75,0.]]}
    def run(phase,name,n):
        if name in r[phase]:return
        seed=258441 if phase=="selection" else 259441
        seeds=np.random.default_rng(seed).bit_generator.random_raw(n)
        w=apply_model("first",variants[name]);parts=[]
        for off in range(0,n,1024):
            p=out/f"{phase}-{name}-{off}.npy"
            if not p.exists():np.save(p,np.asarray(engine.batch_seeds(w,seeds[off:off+1024].tolist(),3)))
            parts.append(np.load(p))
        rows=np.concatenate(parts);hits=set(map(str,seeds[rows[:,6]>0]))
        r[phase][name]=dict(**summarize(rows),heart_encounter_seeds=sorted(hits),
                            victory_seeds=list(map(str,seeds[rows[:,4]>0])))
        if name!="parent":
            old=set(r[phase]["parent"]["heart_encounter_seeds"])
            r[phase][name].update(gained=len(hits-old),lost=len(old-hits))
        write(path,r);print(phase,name,summarize(rows),flush=True)
    # Half-strength / stronger mixtures address a human player's ability to
    # survive the fight they elect to take. Whole-game selection decides that tradeoff.
    screen=("frequency-0.85","context-0.45","context-0.85","context-0.45-camp","context-0.85-camp")
    for name in ("parent",*screen):run("selection",name,8192)
    def rank(k):
        v=r["selection"][k]
        return v["encounters"],v["act4"],v["second"],v["boss"]
    selected=max(screen,key=rank)
    for name in ("parent",selected):run("test",name,32768)
    r["selected"]=selected
    if selected.endswith("-camp"):
        candidate=r["test"][selected]["encounters"];baseline=r["test"]["parent"]["encounters"]
        if candidate>=max(4+baseline,1.25*baseline):
            for name in (selected[:-5],"camp"):run("test",name,32768)
    save_model(out/"challenger.json.gz",variants[selected])
    write(path,r)


if __name__=="__main__":
    main()
