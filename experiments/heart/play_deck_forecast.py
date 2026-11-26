"""Audit the exported afterstate logits, select joint weights, and test ordinary Heart reach."""
import argparse
import json
from pathlib import Path

import heart_native as engine
import numpy as np
import sts_env as sts
import torch

from .deck_outcomes import InventoryForecast
from .heart_curriculum import read_model, save_model
from .replay import apply_model, rows
from .stratified_gate import paired_metric, rank, sampling_design
from .train import write


def audit(parent,params,out):
    w=apply_model("first",parent)
    examples=[];tokens=engine.card_count+engine.relic_count
    kinds=sts.CHOICE_KINDS
    for seed in range(233000000,233000096):
        env=sts.Env();env.reset(seed,20,final_act=True,character="ironclad")
        for action in engine.trace(w,seed):
            o,m=env.observe(),env.mask()
            screen=int(o[6])
            remove=screen==4 and o[26]==4
            if not o[7] and (screen in (2,3) or remove):
                options={};skip=False;eligible=True
                for a in np.flatnonzero(m[81:145])+81:
                    c=rows(o,"choices")[a-81];kind=kinds[int(c[0])]
                    if kind in ("SKIP","SINGING_BOWL"):skip=True
                    elif screen==3 and kind=="RELIC":options[a]=engine.card_count+int(c[1])-1
                    elif screen!=3 and kind=="CARD":options[a]=int(c[1])-1
                    else:eligible=False
                if options and eligible and (skip or remove):
                    inv=np.zeros(tokens,np.float32)
                    for c in rows(o,"deck"):
                        if c[0]:inv[int(c[0])-1]+=1
                    for r in rows(o,"relics"):
                        if r[0]:inv[engine.card_count+int(r[0])-1]+=1
                    examples.append((o.copy(),m.copy(),np.asarray(engine.scores(w,o,m)),options,inv,
                        min(2,int(o[4])-1+(screen==3)),-1 if remove else 1,screen))
            if not env.step(action):raise RuntimeError("parent trace diverged")
    model=InventoryForecast(tokens).eval()
    model.load_state_dict(torch.load(out/"model.pt",weights_only=True))
    scale=torch.tensor([13.,7.,3.])
    engine.set_deck_forecast(*params,scale.tolist())
    worst=0.;counts={}
    with torch.inference_mode():
        for o,m,base,opts,inv,act,sign,screen in examples:
            p=np.stack([inv.copy() for _ in range(1+len(opts))])
            for i,t in enumerate(opts.values(),1):p[i,t]+=sign
            v=model(torch.from_numpy(p),torch.full((len(p),),act),torch.zeros(len(p),6))
            delta=((v[1:]-v[:1])@scale).numpy()
            actual=(np.asarray(engine.scores(w,o,m))-base)[list(opts)]
            worst=max(worst,float(np.max(np.abs(actual-delta))))
            counts[screen]=counts.get(screen,0)+1
    if set(counts)!={2,3,4} or worst>1e-4:raise RuntimeError(f"deck outcome audit failed: {counts}, {worst}")
    return dict(menus_by_screen=counts,max_score_error=worst)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=Path("build/deck-outcomes"));p.add_argument("--threads",type=int,default=4)
    args=p.parse_args();out=args.out;file=out/"gameplay.json"
    r=json.loads(file.read_text()) if file.exists() else {}
    parent=read_model("build/stratified-gate/challenger.json.gz")
    fitted=read_model(out/"fitted.json.gz");params=fitted["deck_outcome"][:-1]
    if "audit" not in r:
        r["audit"]=audit(parent,params,out);write(file,r)
    w=apply_model("first",parent)
    # Oversample earned late sets for selection; population weights restore ordinary frequencies.
    def panel(start,n,name,options):
        path=out/f"scout-{start}.npy"
        engine.reset_deck_forecast()
        if not path.exists():np.save(path,np.asarray(engine.batch(w,start,32768,args.threads)))
        floor=np.load(path)[:,0]
        seeds,strata,sizes=sampling_design(floor,start,n,start+51)
        reference=np.asarray(engine.batch_seeds(w,seeds.tolist(),args.threads))
        candidates=[]
        for weights in options:
            engine.set_deck_forecast(*params,weights)
            data=np.asarray(engine.batch_seeds(w,seeds.tolist(),args.threads))
            candidates.append(dict(params=weights,metrics=paired_metric(data,reference,strata,sizes)))
        candidates.sort(key=rank,reverse=True)
        r[name]=candidates;write(file,r)
    if "training" not in r:
        panel(234000000,2048,"training",[[a,b,c] for a,b,c in (
            (10,5,2),(30,15,5),(60,25,10),(0,25,10),(25,0,8),(40,12,0),(45,0,0),(0,0,15),(0,40,0))])
    if "selection" not in r:
        panel(235000000,4096,"selection",[v["params"] for v in r["training"][:4]])
    weights=r["selection"][0]["params"]
    chosen={**parent,"deck_outcome":[*params,weights]};save_model(out/"challenger.json.gz",chosen)
    ids=np.arange(236000000,236032768);r.setdefault("test",{})
    for name,m in (("parent",parent),("joint",chosen)):
        if name in r["test"]:continue
        w=apply_model("first",m)
        data=np.concatenate([np.asarray(engine.batch_seeds(w,chunk.tolist(),args.threads)) for chunk in np.array_split(ids,16)])
        np.save(out/f"test-{name}.npy",data);hits=data[:,6]>0
        r["test"][name]=dict(runs=len(ids),act3=int((data[:,1]>=3).sum()),act4=int((data[:,1]>=4).sum()),
            first_boss=int((data[:,0]>=50).sum()),second_boss=int((data[:,0]>=51).sum()),
            heart_encounter_seeds=ids[hits].tolist(),victory_seeds=ids[data[:,4]>0].tolist(),truncated=int(data[:,5].sum()))
        write(file,r);print(name,r["test"][name],flush=True)


if __name__=="__main__":
    main()
