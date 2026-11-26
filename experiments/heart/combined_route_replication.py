"""Replicate the learned router on the combined human draft and searches, with a campfire ablation."""
import argparse
import json
from pathlib import Path

import heart_native as engine
import numpy as np

from .heart_curriculum import read_model,save_model
from .replay import apply_model
from .route_rollouts import summarize
from .train import write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm",choices=("route","route_fire"))
    parser.add_argument("--threads",type=int,default=4)
    args=parser.parse_args()
    out=Path("build/combined-route-replication")/args.arm;out.mkdir(exist_ok=True,parents=True)
    model=read_model("build/route-search-transfer/joint/map.json.gz")
    if args.arm=="route_fire":
        model["resource_policy"]=read_model("build/human-macro-replication/fire.json.gz")["resource_policy"]
    w=apply_model("first",model);save_model(out/"model.json.gz",model)
    start=216000000;parts=[]
    for off in range(0,32768,256):
        f=out/f"part-{off}.npy"
        if not f.exists():np.save(f,np.asarray(engine.batch_seeds(w,list(range(start+off,start+off+256)),args.threads)))
        parts.append(np.load(f))
    a=np.concatenate(parts)
    parent=json.loads(Path("build/joint-search-replication/replication/report.json").read_text())["result"]
    baseline=set(map(int,parent["heart_encounter_seeds"]));hits=set((start+np.flatnonzero(a[:,6]>0)).tolist())
    report=dict(**summarize(a),start=start,parent_encounters=len(baseline),parent_wins=parent["hearts"],
                gained=len(hits-baseline),lost=len(baseline-hits),heart_encounter_seeds=sorted(hits),
                victory_seeds=(start+np.flatnonzero(a[:,4]>0)).tolist())
    write(out/"report.json",report);print(args.arm,report,flush=True)


if __name__=="__main__":main()
