"""Fit a set of complementary combat tactics and choose among them by synthetic fights."""
import argparse
from pathlib import Path
from time import perf_counter

import heart_native as engine
import numpy as np

from .deck_training import funnel
from .heart_curriculum import read_model,save_model
from .recovery import panel
from .replay import apply_model
from .train import write


def banks(w,start,threads,n=32768):
    rows=np.asarray(engine.batch(w,start,n,threads))
    return [engine.Bank(w,(start+np.flatnonzero(rows[:,0]>=floor)[:limit]).tolist(),floor,threads)
            for floor,limit in ((15,240),(32,420),(49,250))]


def train_menu(parent,w,positions,threads):
    rng=np.random.default_rng(119041)
    pool=rng.normal(0,rng.uniform(9,39,(84,1)),(84,18))
    pool[:3]=parent["boss_tactics"]
    scores=[]
    def measure(idx):
        parts=[]
        for act,b in enumerate(positions,1):
            r=np.asarray(b.tactic_progress(w,idx,threads))
            if act==3:
                v=.4*(r[:,0]>=51)+.9*(r[:,1]==4)+r[:,2]
            else:
                v=(r[:,1]>act)+.5*(r[:,0]>= (32 if act==1 else 49))+.5*(r[:,1]>act+1)
            parts.append(v/len(r))
        return np.concatenate(parts)
    placeholder=np.zeros((12,18))
    engine.set_scenarios(placeholder.tolist(),1,1)
    reference=measure(0)
    for p in pool:
        placeholder[1]=p
        engine.set_scenarios(placeholder.tolist(),1,1)
        scores.append(measure(1))
    # Add tactics that rescue different training fights. The real selector
    # has to identify the right one using independent hypothetical fights.
    scores=np.stack(scores)
    ceiling=reference.copy()
    chosen=[]
    for _ in range(11):
        gains=np.maximum(scores-ceiling,0).sum(axis=1)
        gains[chosen]=-1
        idx=int(gains.argmax())
        chosen.append(idx)
        ceiling=np.maximum(ceiling,scores[idx])
    return np.vstack([np.zeros(18),pool[chosen]]).tolist()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=Path("build/scenarios"))
    parser.add_argument("--threads",type=int,default=10)
    args=parser.parse_args()
    args.out.mkdir(exist_ok=True,parents=True)
    parent=read_model("build/deck-training/challenger.json.gz")
    w=apply_model("first",parent)
    training=banks(w,119000000,args.threads)
    menu=train_menu(parent,w,training,args.threads)
    print("trained menu on",[len(b) for b in training],"positions",flush=True)
    apply_model("first",parent)
    late=banks(w,120000000,args.threads,16384)[2]
    candidates=[]
    for act,draws in ((1,3),(2,3),(2,8),(3,3),(3,8)):
        params=[menu,draws,act]
        candidate={**parent,"scenarios":params}
        start=perf_counter()
        summary,r=panel(candidate,121000000,2048,args.threads)
        more=np.asarray(late.progress(w,args.threads))
        outcome=dict(act=act,draws=draws,full=summary,late_positions=len(late),
                     late_second=int((more[:,0]>=51).sum()),late_act4=int((more[:,1]==4).sum()),
                     late_heart=int(more[:,2].sum()),seconds=round(perf_counter()-start,2))
        print("validation",act,draws,summary["act3"],outcome["late_heart"],flush=True)
        # Estimate Heart frequency as ordinary boss arrival times conditional
        # late survival, with smoothing because late entries are expensive.
        quality=((r[:,0]>=49).sum()+3)*(1.+outcome["late_heart"]+.3*outcome["late_act4"]+
                                         .05*outcome["late_second"])
        candidates.append((quality,candidate,outcome))
    _,winner,chosen=max(candidates,key=lambda c:c[0])
    save_model(args.out/"challenger.json.gz",winner)
    report=dict(train_start=119000000,training_positions=[len(b) for b in training],
                late_validation_start=120000000,selection_start=121000000,
                candidates=[c[2] for c in candidates],selected=chosen,test_start=122000000,test={})
    for name,model in (("parent",parent),("practice",winner)):
        t=perf_counter()
        stats,r=panel(model,122000000,32768,args.threads)
        report["test"][name]={**stats,**funnel(r),"seconds":round(perf_counter()-t,2)}
        print(name,report["test"][name],flush=True)
    write(args.out/"report.json",report)
    report["confirmation_start"]=123000000
    report["confirmation"]={}
    for name,model in (("parent",parent),("practice",winner)):
        stats,r=panel(model,123000000,32768,args.threads)
        report["confirmation"][name]={**stats,**funnel(r)}
    write(args.out/"verified-panels.json",report)


if __name__=="__main__":
    main()
