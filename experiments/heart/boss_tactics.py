"""Learn separate combat decisions for the three Act 2 bosses from matched fights."""
import argparse
from pathlib import Path

import heart_native as engine
import numpy as np

from .heart_curriculum import read_model, save_model
from .replay import heart_encounters, load_model
from .train import statistics, write


def bank(w,start,limit,threads):
    runs=np.asarray(engine.batch(w,start,8192,threads))
    seeds=start+np.flatnonzero(runs[:,0]>=33)[:limit]
    return engine.Bank(w,seeds.tolist(),33,threads)


def drill(positions,w,params,threads):
    engine.set_boss_tactics(params.tolist())
    data=np.asarray(positions.boss_results(w,threads))
    if (data[:,0]<0).any():
        raise RuntimeError("a floor 33 training position did not enter an Act 2 boss")
    # Keep pressure toward a win even when no sampled policy can finish a fight.
    value=data[:,1]+.25*(1.-data[:,2])+.08*data[:,3]
    ids=data[:,0].astype(int)
    return np.bincount(ids,weights=value)/np.bincount(ids),dict(
        starts=np.bincount(ids).tolist(),clears=np.bincount(ids,weights=data[:,1]).astype(int).tolist())


def fight_headroom(prior,checkpoint,threads):
    w=load_model("first",prior)
    positions=bank(w,107000000,768,threads)
    best=np.asarray(read_model(checkpoint)["boss_tactics"])
    rng=np.random.default_rng(10743)
    variations=rng.normal(size=(96,3,18))*22
    variations[::2]+=best
    choices=np.concatenate(([np.zeros_like(best),best],variations))
    outcomes=[]
    for params in choices:
        engine.set_boss_tactics(params.tolist())
        data=np.asarray(positions.boss_results(w,threads))
        if (data[:,0]<0).any():
            raise ValueError("missing Act 2 boss in headroom audit")
        outcomes.append(data[:,1].astype(bool))
    ids=data[:,0].astype(int)
    result=dict(start=107000000,policies=len(choices),positions=len(positions),by_boss={})
    for name,mask in (("all",ids>=0),("Automaton",ids==0),("Champ",ids==1),("Collector",ids==2)):
        clears=np.stack(outcomes)[:,mask]
        result["by_boss"][name]=dict(entries=int(mask.sum()),forecast=int(clears[0].sum()),
            specialist=int(clears[1].sum()),any_tested_tactics=int(clears.any(axis=0).sum()))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads",type=int,default=10)
    parser.add_argument("--out",type=Path,default=Path("build/boss-tactics"))
    args=parser.parse_args()
    args.out.mkdir(exist_ok=True,parents=True)
    prior=Path("build/forecast/challenger.json.gz")
    w=load_model("first",prior)
    train=bank(w,104000000,768,args.threads)
    validation=bank(w,105000000,512,args.threads)
    centre=np.zeros((3,18))
    spread=np.broadcast_to([22,22,24,28,35,35,18,18,45,30,35,32,15,20,22,30,40,16],centre.shape).copy()
    best=centre.copy()
    best_score,baseline=drill(validation,w,centre,args.threads)
    history=[]
    rng=np.random.default_rng(1322)
    for generation in range(10):
        population=rng.normal(centre,spread,(24,3,18))
        scores=np.stack([drill(train,w,policy,args.threads)[0] for policy in population])
        # One boss's elite policies can differ from another's, even in a
        # generation whose overall best policy is poor against all three.
        top=np.argsort(scores,axis=0)[-6:]
        elites=population[top,np.arange(3)]
        centre=.25*centre+.75*elites.mean(axis=0)
        spread=np.maximum(2.,.35*spread+.65*elites.std(axis=0))
        proposal=elites[-1]
        val,stats=drill(validation,w,proposal,args.threads)
        improved=val>best_score
        best[improved]=proposal[improved]
        best_score=np.maximum(val,best_score)
        history.append(stats)
        print("generation",generation+1,"validation clears",stats["clears"],flush=True)
    _,winner=drill(validation,w,best,args.threads)
    model=read_model(prior)
    model["boss_tactics"]=best.tolist()
    checkpoint=args.out/"challenger.json.gz"
    save_model(checkpoint,model)
    start=106000000
    report=dict(train_start=104000000,validation_start=105000000,test_start=start,
                training_positions=len(train),validation_reference=baseline,validation_specialist=winner,
                bosses=["Automaton","Champ","Collector"],history=history,weights=best.tolist(),test={})
    panels={}
    for name,path in (("forecast",prior),("boss_specialist",checkpoint)):
        weights=load_model("first",path)
        rows=np.asarray(engine.batch(weights,start,32768,args.threads))
        report["test"][name]=dict(**statistics(rows),heart_encounter_seeds=heart_encounters(weights,start,rows))
        panels[name]=rows
        print(name,report["test"][name],flush=True)
    old,new=panels.values()
    if not np.array_equal(old[:,0]>=33,new[:,0]>=33):
        raise AssertionError("policy affected decisions before an Act 2 boss")
    report["boss_entries"]=int((old[:,0]>=33).sum())
    report["gained_act3"]=int(((old[:,1]<3)&(new[:,1]>=3)).sum())
    report["lost_act3"]=int(((old[:,1]>=3)&(new[:,1]<3)).sum())
    report["headroom"]=fight_headroom(prior,checkpoint,args.threads)
    write(args.out/"report.json",report)


if __name__=="__main__":
    main()
