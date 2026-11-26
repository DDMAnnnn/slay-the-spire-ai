"""Select how many public opening-hand scenarios to sample and test the frozen policy."""
import argparse
from pathlib import Path
from time import perf_counter

import heart_native as engine

from .deck_training import funnel
from .heart_curriculum import read_model,save_model
from .recovery import panel
from .replay import apply_model
from .train import write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=Path("build/opening-search"))
    parser.add_argument("--threads",type=int,default=10)
    args=parser.parse_args()
    args.out.mkdir(exist_ok=True,parents=True)
    parent=read_model("build/route-policy/challenger.json.gz")
    candidates=[]
    for draws,scope,scale in ((2,0,2),(2,1,2),(2,2,2),(4,0,6),(4,1,6),(4,2,6),(6,1,15),(6,2,15)):
        model={**parent,"opening_search":[draws,scope,scale]}
        s,r=panel(model,133000000,1536,args.threads)
        rank=(len(s["heart_encounter_seeds"]),s["act4"],int((r[:,0]>=50).sum()),s["act3"])
        candidates.append((rank,model,{**s,**funnel(r),"settings":[draws,scope,scale]}))
    options=[]
    for _,model,_ in sorted(candidates,key=lambda c:c[0],reverse=True)[:3]:
        s,r=panel(model,134000000,3072,args.threads)
        rank=(len(s["heart_encounter_seeds"]),s["act4"],int((r[:,0]>=50).sum()),s["act3"])
        options.append((rank,model,{**s,**funnel(r),"settings":model["opening_search"]}))
    _,best,_=max(options,key=lambda c:c[0])
    save_model(args.out/"challenger.json.gz",best)
    w=apply_model("first",best)
    audit=engine.opening_audit(w,135000000,160)
    print("settings",best["opening_search"],"audit",audit,flush=True)
    report=dict(validation_start=133000000,selection_start=134000000,audit_start=135000000,audit_runs=160,
                eligible_states=audit[0],matched_states=audit[1],accepted_draws=audit[2],
                changed_actions=audit[3],candidates=[c[2] for c in candidates],finalists=[c[2] for c in options],
                selected=best["opening_search"],test={})
    for start in (136000000,137000000):
        report["test"][start]={}
        for name,model in (("route_parent",parent),("opening_search",best)):
            t=perf_counter()
            s,r=panel(model,start,32768,args.threads)
            report["test"][start][name]={**s,**funnel(r),"seconds":round(perf_counter()-t,2)}
            print(start,name,report["test"][start][name],flush=True)
        write(args.out/"report.json",report)


if __name__=="__main__":
    main()
