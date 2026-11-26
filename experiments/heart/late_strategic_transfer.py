"""Deploy the strategic apprentice only after the incumbent has earned the target act."""
from pathlib import Path
import json

import heart_native as engine
import numpy as np

from .heart_curriculum import read_model,save_model
from .replay import apply_model
from .route_rollouts import summarize
from .train import write


def main():
    out=Path("build/late-strategic-transfer");out.mkdir(exist_ok=True,parents=True)
    path=out/"report.json"
    report=json.loads(path.read_text()) if path.exists() else {"selection":{},"test":{}}
    base=read_model("build/human-macro-replication/fire.json.gz")
    weights=read_model("build/strategic-apprentice/apprentice.json.gz")["strategic_apprentice"][:6]
    edits=[0.,0.,.70,.70,.70,0.,.70,.70,.70,0.]
    candidates={"act2-all":(2,[.65]*10),"act3-all":(3,[.65]*10),
                "act2-inventory":(2,edits),"act3-inventory":(3,edits),"act3-full":(3,[1.]*10)}
    def run(phase,name):
        if name in report[phase]:return
        seed=264771 if phase=="selection" else 265771
        ids=np.random.default_rng(seed).bit_generator.random_raw(32768)
        if name=="parent":
            ix=np.arange(len(ids));model=base
        else:
            parent=np.load(out/f"{phase}-parent.npy")
            first,mix=candidates[name];ix=np.flatnonzero(parent[:,1]>=first)
            model={**base,"strategic_apprentice":[*weights,mix,16-(1<<(first-1))]}
        w=apply_model("first",model);parts=[]
        for off in range(0,len(ix),512):
            f=out/f"{phase}-{name}-{off}.npy"
            if not f.is_file():np.save(f,np.asarray(engine.batch_seeds(w,ids[ix[off:off+512]].tolist(),4)))
            parts.append(np.load(f))
        selected=np.concatenate(parts)
        if name=="parent":
            result=selected;np.save(out/f"{phase}-parent.npy",result)
        else:
            result=parent.copy();result[ix]=selected
            if not np.array_equal(result[:,1]>=first,parent[:,1]>=first):raise ValueError("changed frozen prefix")
        hits=set(map(str,ids[result[:,6]>0]))
        info=dict(**summarize(result),eligible=len(ix),heart_encounter_seeds=sorted(hits),
                  victory_seeds=list(map(str,ids[result[:,4]>0])))
        if name!="parent":
            old=set(report[phase]["parent"]["heart_encounter_seeds"]);info.update(gained=len(hits-old),lost=len(old-hits))
        report[phase][name]=info;write(path,report);print(phase,name,summarize(result),flush=True)
        if phase=="test" and name!="parent":save_model(out/"selected.json.gz",model)
    for name in ("parent",*candidates):run("selection",name)
    def rank(k):
        s=report["selection"][k];return s["encounters"],s["wins"],s["act4"],s["second"]
    best=max(candidates,key=rank);report["selected"]=best
    for name in ("parent",best):run("test",name)
    write(path,report)


if __name__=="__main__":main()
