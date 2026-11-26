"""Compare visible-map imitation with public model-based planning over the next rooms of Act 3."""
import argparse
import json
from pathlib import Path

import heart_native as engine
import numpy as np

from .heart_curriculum import read_model,save_model
from .replay import apply_model
from .train import write


def summarize(a):
    return dict(n=len(a),encounters=int(a[:,6].sum()),wins=int(a[:,4].sum()),act3=int((a[:,1]>=3).sum()),
                boss=int((a[:,0]>=50).sum()),second=int((a[:,0]>=51).sum()),act4=int((a[:,1]>=4).sum()),truncated=int(a[:,5].sum()))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--threads",type=int,default=3);args=p.parse_args()
    out=Path("build/route-rollouts");out.mkdir(parents=True,exist_ok=True);path=out/"report.json"
    report=json.loads(path.read_text()) if path.exists() else {"selection":{}}
    parent=read_model("build/route-irl/selected-whole_map.json.gz");w=apply_model("first",parent)
    def scout(start):
        path=out/f"scout-{start}.npy"
        if not path.exists():
            engine.set_route_worlds(0,4,3,0)
            np.save(path,np.concatenate([np.asarray(engine.batch_seeds(w,list(range(start+i,start+i+1024)),args.threads))
                                          for i in range(0,32768,1024)]))
        return np.load(path)
    start=249000000;ref=scout(start)
    ix=np.random.default_rng(24937).choice(np.flatnonzero(ref[:,1]>=3),384,replace=False)
    configs=((3,4,3,50.),(6,4,3,100.),(6,8,3,60.),(10,8,3,120.),(6,13,3,100.))
    if "audit" not in report:
        engine.set_route_worlds(*configs[0])
        audit=list(engine.route_world_audit(w,(start+ix[:32]).tolist()))
        if not audit[0] or not audit[2]:raise ValueError(f"no usable or changed route from the public roots: {audit}")
        report["audit"]=dict(seeds=32,searched=audit[0],accepted_worlds=audit[1],changed=audit[2]);write(path,report)
    report["selection_control"]=summarize(ref[ix])
    for k,cfg in enumerate(configs):
        if str(k) in report["selection"]:continue
        engine.set_route_worlds(*cfg)
        v=np.asarray(engine.batch_seeds(w,(start+ix).tolist(),args.threads))
        report["selection"][str(k)]=dict(settings=cfg,**summarize(v));write(path,report)
        print("selection",k,report["selection"][str(k)],flush=True)
    def rank(x):return (x["encounters"],x["act4"],x["second"],x["boss"])
    winner=max(report["selection"].values(),key=rank)
    start=250000000;ref=scout(start);eligible=np.flatnonzero(ref[:,1]>=3)
    cfg=winner["settings"];engine.set_route_worlds(*cfg)
    candidate={**parent,"route_worlds":cfg};save_model(out/"challenger.json.gz",candidate)
    result=ref.copy()
    for offset in range(0,len(eligible),64):
        ids=eligible[offset:offset+64];p=out/f"test-{offset}.npy"
        if not p.exists():np.save(p,np.asarray(engine.batch_seeds(w,(start+ids).tolist(),args.threads)))
        result[ids]=np.load(p)
    if not np.array_equal(result[:,1]>=3,ref[:,1]>=3):raise RuntimeError("changed game before Act 3 entry")
    a,b=result[:,6]>0,ref[:,6]>0
    report["test"]=dict(start=start,parent=summarize(ref),candidate=summarize(result),eligible=len(eligible),
        settings=cfg,gained=int((a&~b).sum()),lost=int((b&~a).sum()),heart_encounter_seeds=(start+np.flatnonzero(a)).tolist(),
        parent_encounter_seeds=(start+np.flatnonzero(b)).tolist(),victory_seeds=(start+np.flatnonzero(result[:,4])).tolist())
    write(path,report);print("ordinary",report["test"],flush=True)


if __name__=="__main__":
    main()
