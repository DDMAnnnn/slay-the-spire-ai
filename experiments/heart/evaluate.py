"""Evaluate a frozen Heart policy on a predeclared consecutive range without retaining observations."""
import argparse
import json
from pathlib import Path

import numpy as np
import heart_native as engine
from .train import configure, verify, write


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
        write(out/"synergies.json",args.variant=="synergies")
        configure(out,out)
    else:
        configure(args.model.parent,out)
        w=np.asarray(json.loads(args.model.read_text())["weights"])
    counts=np.zeros(6,dtype=np.int64)
    max_floor=0
    with (out/"batches.jsonl").open("w",encoding="utf8") as log:
        for offset in range(0,args.count,1024):
            n=min(1024,args.count-offset)
            rows=np.asarray(engine.batch(w.tolist(),args.start+offset,n,args.threads))
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
                        hearts=int(counts[4]),truncated=int(counts[5]))
            log.write(json.dumps(report)+"\n")
            log.flush()
            if offset%16384==0 or total==args.count:
                print(report,flush=True)
    write(out/"summary.json",report)


if __name__=="__main__":
    main()
