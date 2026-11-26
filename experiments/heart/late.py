"""Specialize policy weights on natural Act 2 and 3 starts, using separate validation banks."""
import argparse
import json
from pathlib import Path
import shutil

import heart_native as engine
import numpy as np
from .train import SCALE, write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("prior",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("--threads",type=int,default=6)
    p.add_argument("--generations",type=int,default=14)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    policy=json.loads(args.prior.read_text())
    base=np.array(policy["weights"])
    for filename,setter in (("event-table",engine.set_event_table),("choice-table",engine.set_choice_table),
                            ("combat-head",lambda h:engine.set_head(*h))):
        src=args.prior.parent/(filename+".json")
        if src.exists():
            setter(json.loads(src.read_text()))
            shutil.copyfile(src,args.out/src.name)
    def make(seed):
        return [(engine.Bank(base.tolist(),seed+act*90000,64000,n,act),act)
                for act,n in ((2,240),(3,192))]
    train=make(5000000)
    validation=make(5700000)
    print("Bank sizes",[(len(b),a) for b,a in train],[(len(b),a) for b,a in validation],flush=True)
    def value(w,banks):
        engine.set_late(2,w.tolist())
        score=0.
        for b,act in banks:
            rows=np.asarray(b.batch(base.tolist(),args.threads))
            if len(rows)==0:
                raise RuntimeError("No naturally reached curriculum states")
            floor=rows[:,0]
            score+=np.mean(floor/12 + .9*(rows[:,1]>act) + .55*(floor>=49) +
                           .8*(floor>=51) + 3*(rows[:,1]==4) + 18*rows[:,4] - rows[:,5])
        return float(score)
    rng=np.random.default_rng(82970)
    best=base.copy()
    centre=best.copy()
    best_valid=value(best,validation)
    history=[]
    for gen in range(args.generations):
        noise=rng.normal(size=(8,len(base)))
        population=np.maximum(.02,centre+np.vstack([noise,-noise])*SCALE*(.96**gen)*1.6)
        scores=[value(x,train) for x in population]
        winners=np.argsort(scores)[-4:]
        centre=.3*centre+.7*population[winners].mean(axis=0)
        for candidate in (centre,population[winners[-1]]):
            metric=value(candidate,validation)
            if metric>best_valid:
                best=candidate.copy()
                best_valid=metric
        history.append(dict(generation=gen+1,validation=best_valid,training=max(scores)))
        write(args.out/"late.json",dict(act=2,weights=best.tolist(),history=history))
        write(args.out/"policy.json",policy)
        print(history[-1],flush=True)


if __name__=="__main__":
    main()
