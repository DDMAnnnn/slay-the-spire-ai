"""Compare a deck-conditioned value of taking a card with a per-card mean and the previous policy."""
import argparse
import gzip
import json
from pathlib import Path

import heart_native as engine
import numpy as np

from .boss_relics import compare
from .preferences import fit_item_values
from .replay import heart_encounters, load_model, set_draft
from .train import statistics, write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prior",type=Path,default=Path("build/first-boss-relics/challenger.json.gz"))
    p.add_argument("--out",type=Path,default=Path("build/context-draft"))
    p.add_argument("--threads",type=int,default=10)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    w=load_model("first",args.prior)
    data=np.asarray(engine.draft_practice(w,68000000,8192,args.threads))
    if data.size==0:
        raise RuntimeError("No card rewards sampled")
    fitted=fit_item_values(data,engine.card_count)
    print(f"Collected {len(data)} alternatives",flush=True)
    ids=data[:,0].astype(int)
    mean=np.zeros_like(fitted)
    mean[:,0]=np.bincount(ids,weights=data[:,-1],minlength=len(mean))/(32+np.bincount(ids,minlength=len(mean)))
    report=dict(train_start=68000000,train_runs=8192,examples=len(data),validation_start=69000000,
                validation={},test_start=70000000,test={})
    families={}
    for name,base in (("id_only",mean),("context",fitted)):
        candidates={}
        for cost in (-1.,0.,1.):
            table=base.copy()
            table[:,0]-=cost
            set_draft(table.tolist())
            candidates[cost]=statistics(np.asarray(engine.batch(w,69000000,4096,args.threads)))
        selected=max(candidates,key=lambda c:(candidates[c]["act3"],candidates[c]["mean_floor"]))
        report["validation"][name]=dict(candidates=candidates,cost=selected)
        table=base.copy()
        table[:,0]-=selected
        families[name]=table.tolist()
    with gzip.open(args.prior,"rt",encoding="utf8") as f:
        model=json.load(f)
    for name,filename in (("context","challenger.json.gz"),("id_only","id-only.json.gz")):
        model["draft_context"]=families[name]
        with gzip.open(args.out/filename,"wt",encoding="utf8") as f:
            json.dump(model,f)
    for name,table in {"previous":None,**families}.items():
        set_draft(table)
        result=np.asarray(engine.batch(w,70000000,32768,args.threads))
        if name=="previous":
            previous=result
        report["test"][name]=dict(**statistics(result),**compare(previous,result),
                                 heart_encounter_seeds=heart_encounters(w,70000000,result))
        print(name,report["test"][name],flush=True)
    write(args.out/"report.json",report)


if __name__=="__main__":
    main()
