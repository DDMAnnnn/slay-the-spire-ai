"""Evaluate a frozen Heart policy on a predeclared consecutive range without retaining observations."""
import argparse
import json
from pathlib import Path

import numpy as np
import heart_native as engine
from .train import configure, verify, write
from .replay import heart_encounters


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("model",type=Path)
    p.add_argument("start",type=int)
    p.add_argument("count",type=int)
    p.add_argument("--threads",type=int,default=6)
    p.add_argument("--variant",choices=("first","synergies"),default="first",help="variant for a packaged checkpoint")
    p.add_argument("--out",type=Path,help="output directory for batch logs")
    args=p.parse_args()
    if args.count<=0 or args.threads<=0:
        p.error("count and threads must be positive")
    out=args.out or (Path("build") / f"heart-{args.variant}-{args.start}-{args.count}" if args.model.suffix==".gz" else args.model.parent / f"panel-{args.start}-{args.count}")
    out.mkdir(parents=True,exist_ok=True)
    if args.model.suffix == ".gz":
        import gzip
        with gzip.open(args.model,"rt",encoding="utf8") as f:
            model=json.load(f)
        w=np.asarray(model["weights"])
        for name, key in (("event-table","events"),("choice-table","choices"),("combat-head","combat"),("late","late")):
            write(out/(name+".json"),model[key])
        write(out/"synergies.json",model.get("policy_synergies",args.variant=="synergies"))
        write(out/"first-boss-relics.json",model.get("first_boss_relics", [0.] * engine.relic_count))
        write(out/"turn-depth.json",model.get("turn_depth", 2))
        write(out/"combat-preferences.json",model.get("combat_preferences", {}))
        write(out/"neow.json",model.get("neow"))
        write(out/"draft-context.json",model.get("draft_context"))
        write(out/"late-combat.json",model.get("late_combat"))
        write(out/"late-combat-last-act.json",model.get("late_combat_last_act",4))
        write(out/"final-weights.json",model.get("final_weights"))
        write(out/"direct-actor.json",model.get("direct_actor"))
        write(out/"forecast.json",model.get("forecast"))
        write(out/"boss-tactics.json",model.get("boss_tactics"))
        write(out/"recovery.json",model.get("recovery"))
        write(out/"deck-edits.json",model.get("deck_edits"))
        write(out/"scenarios.json",model.get("scenarios"))
        write(out/"inventory-value.json",model.get("inventory_value"))
        write(out/"curriculum-actor.json",model.get("curriculum_actor"))
        write(out/"expert-prior.json",model.get("expert_prior"))
        write(out/"boss-draft.json",model.get("boss_draft",[]))
        write(out/"trajectory-teacher.json",model.get("trajectory_teacher"))
        write(out/"route.json",model.get("route"))
        write(out/"opening-search.json",model.get("opening_search"))
        configure(out,out)
    else:
        configure(args.model.parent,out)
        w=np.asarray(json.loads(args.model.read_text())["weights"])
    counts=np.zeros(6,dtype=np.int64)
    encounters=0
    max_floor=0
    with (out/"batches.jsonl").open("w",encoding="utf8") as log:
        for offset in range(0,args.count,1024):
            n=min(1024,args.count-offset)
            rows=np.asarray(engine.batch(w.tolist(),args.start+offset,n,args.threads))
            encounters+=len(heart_encounters(w.tolist(),args.start+offset,rows))
            for index in np.flatnonzero(rows[:,4]):
                if not counts[4]:
                    verify(w,args.start+offset+int(index),out/"heart-win.json")
                counts[4]+=1
            counts[0]+=rows[:,0].sum()
            for act in (2,3,4):
                counts[act-1]+=(rows[:,1]>=act).sum()
            counts[5]+=rows[:,5].sum()
            max_floor=max(max_floor,int(rows[:,0].max()))
            total=offset+n
            report=dict(start=args.start,count=total,planned=args.count,mean_floor=counts[0]/total,
                        max_floor=max_floor,act2=int(counts[1]),act3=int(counts[2]),act4=int(counts[3]),
                        hearts=int(counts[4]),heart_encounters=encounters,truncated=int(counts[5]))
            log.write(json.dumps(report)+"\n")
            log.flush()
            if offset%16384==0 or total==args.count:
                print(report,flush=True)
    write(out/"summary.json",report)


if __name__=="__main__":
    main()
