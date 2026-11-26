"""Fit inventory decision values from paired changes made before each act's boss."""
import argparse
from pathlib import Path

import heart_native as engine
import numpy as np

from .heart_curriculum import read_model, save_model
from .recovery import panel
from .replay import apply_model
from .train import write


def funnel(rows):
    return dict(act3=int((rows[:,1]>=3).sum()),first_boss=int((rows[:,0]>=50).sum()),
                second_boss=int((rows[:,0]>=51).sum()),act4=int((rows[:,1]>=4).sum()))


def fit(data):
    size=engine.card_count
    weights=np.zeros((3,4,size,4))
    penalty=np.diag([18,70,70,90])
    for op in range(4):
        edits=data[data[:,1]==op]
        for id in np.unique(edits[:,2]).astype(int):
            rows=edits[edits[:,2]==id]
            x=np.column_stack([np.ones(len(rows)),rows[:,3:6]])
            target=rows[:,-1]
            shared=np.linalg.solve(x.T@x+penalty,x.T@target)
            for act in range(3):
                mask=rows[:,0]==act
                xa=x[mask]
                # Shrink rare Act 3 edits toward the same item's pooled effect.
                weights[act,op,id]=np.linalg.solve(xa.T@xa+penalty,
                    xa.T@target[mask]+penalty@shared)
    return weights


def collect(w,start,rows,threads):
    data=[]
    visits={}
    for floor,limit in ((15,1600),(32,1800),(49,768)):
        seeds=start+np.flatnonzero(rows[:,0]>=floor)[:limit]
        count=0
        for k in range(0,len(seeds),128):
            bank=engine.Bank(w,seeds[k:k+128].tolist(),floor,threads)
            batch=np.asarray(bank.edit_practice(w,start+50000*floor+k,48 if floor==49 else 24,threads))
            if len(batch):
                data.append(batch)
                count+=len(batch)
        print("floor",floor,"entries",len(seeds),"matched edits",count,flush=True)
        visits[floor]=dict(entries=len(seeds),paired_edits=count)
    return np.concatenate(data),visits


def followup(parent,winner,threads):
    report={"confirmation":{},"components":{}}
    for name,model in (("parent",parent),("edits",winner)):
        stats,r=panel(model,118000000,32768,threads)
        report["confirmation"][name]={**stats,**funnel(r)}
        print("confirmation",name,report["confirmation"][name],flush=True)
    weights=np.asarray(winner["deck_edits"])
    for op,name in enumerate(("add","upgrade","remove","potion")):
        partial=np.zeros_like(weights)
        partial[:,op]=weights[:,op]
        stats,r=panel({**parent,"deck_edits":partial.tolist()},117000000,32768,threads)
        report["components"][name]={**stats,**funnel(r)}
        print("component",name,stats["act3"],stats["heart_encounter_seeds"],flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=Path("build/deck-training"))
    p.add_argument("--threads",type=int,default=10)
    args=p.parse_args()
    args.out.mkdir(exist_ok=True,parents=True)
    parent=read_model("build/recovery/draft-only.json.gz")
    w=apply_model("first",parent)
    train=np.asarray(engine.batch(w,113000000,32768,args.threads))
    data,visits=collect(w,113000000,train,args.threads)
    if set(data[:,0].astype(int))!={0,1,2}:
        raise RuntimeError("missing a training act")
    np.save(args.out/"samples.npy",data)
    table=fit(data)
    valid=np.asarray(engine.batch(w,114000000,16384,args.threads))
    # A natural Act 3 entry panel gives a second estimate for later actions
    # when a small whole-game validation panel yields zero Heart encounters.
    bank=engine.Bank(w,(114000000+np.flatnonzero(valid[:,0]>=35)[:500]).tolist(),35,args.threads)
    rng=np.random.default_rng(11349)
    variants=[]
    for i in range(24):
        multiplier=np.exp(rng.uniform(np.log(15.),np.log(240.),4))
        transfer=rng.uniform(0,.8)
        weights=table.copy()
        for act in (0,1):
            weights[act]=(1-transfer)*table[act]+transfer*table[act+1]
        weights*=multiplier[None,:,None,None]
        weights=weights.tolist()
        engine.set_edits(weights)
        rows=np.asarray(engine.batch(w,115000000,768,args.threads))
        late=np.asarray(bank.progress(w,args.threads))
        rank=(np.log(.01+(rows[:,1]>=3).mean())+np.log(.02+(late[:,0]>=50).mean())
              +np.log(.007+.4*(late[:,0]>=51).mean()+.6*(late[:,1]>=4).mean()+late[:,2].mean()))
        variants.append((rank,weights,multiplier.tolist(),float(transfer)))
    candidates=[]
    for _,weights,multipliers,transfer in sorted(variants,key=lambda r:r[0],reverse=True)[:3]:
        model={**parent,"deck_edits":weights}
        result,_=panel(model,116000000,8192,args.threads)
        candidates.append((model,result,multipliers,transfer))
    best=max(candidates,key=lambda c:(len(c[1]["heart_encounter_seeds"]),c[1]["act4"],c[1]["act3"]))
    save_model(args.out/"challenger.json.gz",best[0])
    report=dict(train_start=113000000,training=visits,training_funnel=funnel(train),
                distribution=np.bincount(data[:,1].astype(int)).tolist(),search_size=len(variants),
                late_validation_start=114000000,late_validation_size=len(bank),
                selection_start=116000000,selection_runs=8192,test_start=117000000,
                validation=[dict(summary=r,multipliers=m,transfer=t) for _,r,m,t in candidates],
                selected=dict(multipliers=best[2],transfer=best[3]),test={})
    for name,model in (("parent",parent),("edits",best[0])):
        stats,rows=panel(model,117000000,32768,args.threads)
        report["test"][name]={**stats,**funnel(rows)}
        np.save(args.out/(name+"-test.npy"),rows)
        print(name,report["test"][name],flush=True)
    write(args.out/"report.json",report)
    if report["test"]["edits"]["act3"]>1.1*report["test"]["parent"]["act3"]:
        write(args.out/"followup.json",followup(parent,best[0],args.threads))


if __name__=="__main__":
    main()
