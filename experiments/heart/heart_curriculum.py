"""Optimize late-run weights from naturally reached Act 3 and pre-boss positions."""
import argparse
import gzip
import json
from pathlib import Path

import heart_native as engine
import numpy as np

from .replay import heart_encounters, load_model
from .train import SCALE, statistics, write


def read_model(path):
    with gzip.open(path,"rt",encoding="utf8") as f:
        return json.load(f)


def save_model(path,model):
    with gzip.open(path,"wt",encoding="utf8") as f:
        json.dump(model,f)


def banks(w,start,n,threads):
    rows=np.asarray(engine.batch(w,start,n,threads))
    result=[]
    for floor,limit in ((35,512),(49,384)):
        ids=(start+np.flatnonzero(rows[:,0]>=floor)[:limit]).tolist()
        result.append(engine.Bank(w,ids,floor,threads))
    print("Floor 35 and 49 sample sizes", [len(b) for b in result],flush=True)
    return result


def quality(w,weights,banks,threads):
    engine.set_final_weights(weights.tolist())
    score=0.
    stats=[]
    for bank in banks:
        rows=np.asarray(bank.progress(w,threads))
        # Both panels contribute equally. Floors and boss clears provide gradients
        # between rare encounters, which receive the largest individual credit.
        score+=(rows[:,0]/16+1.*(rows[:,0]>=51)+2.*(rows[:,1]==4)+12.*rows[:,2]-rows[:,4]).mean()
        stats.append(dict(n=len(rows),floor=float(rows[:,0].mean()),act4=int((rows[:,1]==4).sum()),
                          encounters=int(rows[:,2].sum()),wins=int(rows[:,3].sum())))
    return float(score),stats


def evaluate(reference,staged,challenger,threads,out,report):
    for name,path in (("reference",reference),("staged",staged),("curriculum",challenger)):
        w=load_model("first",path)
        result=np.asarray(engine.batch(w,80000000,32768,threads))
        summary=dict(**statistics(result),heart_encounter_seeds=heart_encounters(w,80000000,result))
        if name=="staged":
            before=result
        if name=="curriculum":
            if not np.array_equal(before[:,1]>=3,result[:,1]>=3):
                raise RuntimeError("late-only weights changed which seeds reached Act 3")
        report["test"][name]=summary
        print(name,summary,flush=True)
        write(out/"report.json",report)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference",type=Path,default=Path("build/first-boss-relics/challenger.json.gz"))
    p.add_argument("--draft",type=Path,default=Path("build/context-draft/id-only.json.gz"))
    p.add_argument("--specialist",type=Path,default=Path("build/late-combat-hand/challenger.json.gz"))
    p.add_argument("--threads",type=int,default=10)
    p.add_argument("--training-starts",type=int,default=32768,help="ordinary starts for finding training positions")
    p.add_argument("--out",type=Path,default=Path("build/heart-curriculum"))
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    model=read_model(args.draft)
    model.update(late_combat=read_model(args.specialist)["late_combat"],late_combat_last_act=2)
    staged=args.out/"staged.json.gz"
    save_model(staged,model)
    w=load_model("first",staged)
    train=banks(w,78000000,args.training_starts,args.threads)
    valid=banks(w,79000000,32768,args.threads)
    rng=np.random.default_rng(780519)
    centre=np.array(model["late"]["weights"])
    best=centre.copy()
    initial,initial_stats=quality(w,best,valid,args.threads)
    best_value=initial
    history=[]
    for gen in range(12):
        noise=rng.normal(size=(10,len(centre)))
        population=np.maximum(.02,centre+np.vstack([noise,-noise])*SCALE*(2.4*.94**gen))
        scores=[quality(w,trial,train,args.threads)[0] for trial in population]
        elite=np.argsort(scores)[-4:]
        centre=.3*centre+.7*population[elite].mean(axis=0)
        for candidate in (centre,population[elite[-1]]):
            value,_=quality(w,candidate,valid,args.threads)
            if value>best_value:
                best_value,best=value,candidate.copy()
        history.append(dict(generation=gen+1,train=max(scores),validation=best_value))
        print(history[-1],flush=True)
    _,fit_stats=quality(w,best,valid,args.threads)
    model["final_weights"]=best.tolist()
    challenger=args.out/"challenger.json.gz"
    save_model(challenger,model)
    report=dict(training_start=78000000,training_starts=args.training_starts,validation_start=79000000,validation_prior=initial_stats,
                validation_trained=fit_stats,history=history,test_start=80000000,test={})
    write(args.out/"report.json",report)
    evaluate(args.reference,staged,challenger,args.threads,args.out,report)


if __name__=="__main__":
    main()
