"""Evaluate independent, public-observation combat particles against ordinary A20 starts."""
import argparse
from pathlib import Path
from time import perf_counter

import heart_native as engine
import numpy as np

from .deck_training import funnel
from .heart_curriculum import read_model,save_model
from .replay import apply_model,heart_encounters
from .train import statistics,write


def evaluate(weights,start,n,threads):
    t=perf_counter()
    rows=np.asarray(engine.batch(weights,start,n,threads))
    return {**statistics(rows),**funnel(rows),"heart_encounter_seeds":heart_encounters(weights,start,rows),
            "seconds":round(perf_counter()-t,2)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=Path("build/public-belief"))
    p.add_argument("--threads",type=int,default=8)
    p.add_argument("--pilot",action="store_true")
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    parent=read_model("build/trajectory-teacher/challenger.json.gz")
    w=apply_model("first",parent)
    count=384 if args.pilot else 3072
    engine.collect_beliefs(w,190000000,count)
    engine.set_belief(1,2,2.)
    audit=engine.belief_audit(w,191000000,40)
    print("eligible, enemy match, full public match, particles, changed",audit,flush=True)
    runs=128 if args.pilot else 1024
    settings=((0,0,0.),(1,0,2.),(1,2,4.)) if args.pilot else (
        (0,0,0.),(1,0,2.),(1,2,3.),(2,2,2.),(3,1,8.),(4,0,10.))
    report=dict(bank_start=190000000,bank_seeds=count,audit=list(audit),validation=[],test={})
    for draws,scope,scale in settings:
        engine.set_belief(draws,scope,scale)
        result=evaluate(w,191000000,runs,args.threads)
        result.update(draws=draws,scope=scope,scale=scale)
        report["validation"].append(result)
        print("screen",result,flush=True)
        write(args.out/"screen.json",report)
    if args.pilot:return
    def rank(v):
        return v["first_boss"]+2*v["second_boss"]+.1*v["act3"]+4*v["act4"]
    chosen=max(report["validation"][1:],key=rank)
    cfg=dict(seed=190000000,count=count,draws=chosen["draws"],scope=chosen["scope"],scale=chosen["scale"])
    save_model(args.out/"challenger.json.gz",{**parent,"public_belief":cfg})
    report["selected"]=cfg
    for name,settings in (("parent",(0,0,0.)),("belief",(cfg["draws"],cfg["scope"],cfg["scale"]))):
        engine.set_belief(*settings)
        report["test"][name]=evaluate(w,192000000,32768,args.threads)
        print("test",name,report["test"][name],flush=True)
        write(args.out/"report.json",report)


if __name__=="__main__":
    main()
