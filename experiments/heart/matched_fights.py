"""Calibrate stronger public search on naturally lost fights and paired winning controls."""
import argparse
from pathlib import Path
from time import perf_counter

import heart_native as engine
import numpy as np

from .heart_curriculum import read_model,save_model
from .inventory_search import panel_chunks
from .replay import apply_model
from .train import write


SETTINGS={
    "parent":(0,0,0.,1,32.,False),
    "current_search":(3,1,8.,1,32.,False),
    "late_examples":(3,1,8.,1,32.,True),
    "survival":(6,2,8.,1,90.,True),
    "portfolio":(4,2,8.,3,90.,True),
    "deeper_portfolio":(8,2,12.,3,90.,True)
}


def configure(setting):
    draws,scope,scale,plans,win,enriched=setting
    engine.set_belief(draws,scope,scale)
    engine.set_belief_plans(plans,win,enriched)


def summary(rows):
    meta=rows[:,:6];results=rows[:,6:]
    out={}
    for name,winner in (("fatal",0),("control",1)):
        mask=meta[:,1]==winner
        r=results[mask]
        out[name]=dict(fights=len(r),wins=int(r[:,0].sum()),mean_exit_hp=float(r[:,1].mean()),
                       mean_potions=float(r[:,2].mean()),capped=int(r[:,3].sum()))
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=Path("build/matched-fights"))
    p.add_argument("--threads",type=int,default=4)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    parent=read_model("build/trajectory-teacher/challenger.json.gz")
    w=apply_model("first",parent)
    engine.collect_beliefs(w,190000000,3072)
    extras=engine.enrich_late_particles(w,192)
    report=dict(synthetic_opponents=list(extras),practice_start=201000000,heldout_start=202000000,calibration={},heldout={})
    def assess(start,n,names):
        records={k:[] for k in names}
        timings=dict.fromkeys(names,0.)
        for offset in range(0,n,32):
            configure(SETTINGS["parent"])
            bank=engine.MatchedFights(w,start+offset,min(32,n-offset),args.threads)
            meta=np.asarray(bank.info);meta[:,0]+=offset
            for name in names:
                configure(SETTINGS[name])
                t=perf_counter()
                values=np.asarray(bank.evaluate(w,args.threads))
                timings[name]+=perf_counter()-t
                if name=="parent" and np.any(meta[:,1]!=values[:,0]):
                    raise RuntimeError("natural winning or fatal fight did not reproduce")
                records[name].append(np.column_stack((meta,values)))
        output={}
        for name,r in records.items():
            x=np.concatenate(r)
            np.save(args.out/f"{start}-{name}.npy",x)
            output[name]={**summary(x),"seconds":round(timings[name],2)}
        return output
    report["calibration"]=assess(201000000,160,SETTINGS)
    def rank(name):
        v=report["calibration"][name]
        return v["fatal"]["wins"]-4*(v["control"]["fights"]-v["control"]["wins"])+.1*v["control"]["mean_exit_hp"]
    chosen=max(list(SETTINGS)[2:],key=rank)
    report["chosen"]=chosen
    write(args.out/"calibration.json",report)
    report["heldout"]=assess(202000000,256,("parent","current_search",chosen))
    write(args.out/"matched.json",report)
    print("matched",chosen,report["heldout"],flush=True)
    proposed=report["heldout"][chosen];reference=report["heldout"]["current_search"]
    if all(proposed[k]["wins"]<=reference[k]["wins"] and
           proposed[k]["mean_exit_hp"]<=reference[k]["mean_exit_hp"] for k in ("fatal","control")):
        report["qualification"]="Dominated on survival and exit health in both held-out groups"
        write(args.out/"report.json",report)
        return
    d,scope,scale,plans,win,extra=SETTINGS[chosen]
    cfg=dict(seed=190000000,count=3072,draws=d,scope=scope,scale=scale,plans=plans,win_value=win,enrich=192 if extra else 0)
    save_model(args.out/"challenger.json.gz",{**parent,"public_belief":cfg})
    # Use the same 32,768 starts as the frozen parent and original three-particle search.
    configure(SETTINGS[chosen])
    report["panel_start"]=192000000
    report["test"]=panel_chunks(args.out,"challenger",w,192000000,32768,args.threads)
    write(args.out/"report.json",report)


if __name__=="__main__":
    main()
