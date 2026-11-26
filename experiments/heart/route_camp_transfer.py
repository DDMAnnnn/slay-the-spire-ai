"""Replicate the Act 3 public room planner on starts played with the validated campfire policy."""
import json
from pathlib import Path

import heart_native as engine
import numpy as np

from .heart_curriculum import read_model,save_model
from .replay import apply_model
from .route_rollouts import summarize
from .train import write


def main():
    out=Path("build/route-camp-transfer");out.mkdir(parents=True,exist_ok=True)
    reference=Path("build/human-macro-replication")
    parent=read_model(reference/"fire.json.gz")
    cfg=json.loads(Path("build/route-rollouts/report.json").read_text())["test"]["settings"]
    model={**parent,"route_worlds":cfg};w=apply_model("first",model)
    ids=np.random.default_rng(260117).bit_generator.random_raw(32768)
    ref=np.concatenate([np.load(reference/f"fire-{i}.npy") for i in range(0,len(ids),1024)])
    eligible=np.flatnonzero(ref[:,1]>=3);result=ref.copy()
    for i in range(0,len(eligible),48):
        part=eligible[i:i+48];f=out/f"part-{i}.npy"
        if not f.exists():np.save(f,np.asarray(engine.batch_seeds(w,ids[part].tolist(),3)))
        result[part]=np.load(f)
    if not np.array_equal(result[:,1]>=3,ref[:,1]>=3):raise ValueError("earlier act changed")
    a,b=result[:,6]>0,ref[:,6]>0
    r=dict(runs=len(ids),eligible=len(eligible),settings=cfg,parent=summarize(ref),candidate=summarize(result),
           gained=int((a&~b).sum()),lost=int((b&~a).sum()),heart_encounter_seeds=list(map(str,ids[a])),
           victory_seeds=list(map(str,ids[result[:,4]>0])))
    write(out/"report.json",r);save_model(out/"candidate.json.gz",model);print(r,flush=True)


if __name__=="__main__":main()
