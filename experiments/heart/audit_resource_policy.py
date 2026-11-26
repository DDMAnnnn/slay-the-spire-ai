"""Check native and training utilities on actual camp, shop, upgrade and removal menus."""
from collections import Counter
import json
from pathlib import Path

import heart_native as engine
import numpy as np
import sts_env as sts
import torch

from .heart_curriculum import read_model
from .replay import apply_model, rows
from .resource_imitation import ResourcePolicy, pack
from .train import write


def main():
    out=Path("build/resource-policy")
    cfg=read_model(out/"fitted.json.gz")["resource_policy"]
    sizes=json.loads((out/"sizes.json").read_text())
    net=ResourcePolicy(len(cfg[2])).eval()
    net.load_state_dict(torch.load(out/"resources.pt",weights_only=True))
    parent=read_model("build/stratified-gate/challenger.json.gz")
    w=apply_model("first",parent)
    examples=[]
    for seed in range(229000000,229000064):
        game=sts.Env();game.reset(seed,20,final_act=True,character="ironclad")
        for action in engine.trace(w,seed):
            o,mask=game.observe(),game.mask()
            screen=int(o[6]);task=int(o[26])
            if not o[7] and (screen in (7,8) or screen==4 and task in (3,4)):
                kind=0 if screen==7 else 1 if screen==8 else 2 if task==3 else 3
                ids=[];legal=[]
                for a in np.flatnonzero(mask[81:145])+81:
                    typ,item,up,price,*_=map(int,rows(o,"choices")[a-81])
                    if typ==10: option=(2+item,2,0,0)
                    elif typ==1: option=(8+item,3,int(bool(up)),price)
                    elif typ==2: option=(8+sizes["cards"]+item,4,0,price)
                    elif typ==3: option=(8+sizes["cards"]+sizes["relics"]+item,5,0,price)
                    elif typ==11: option=(1,1,0,price)
                    elif typ==6: option=(0,0,0,0)
                    else: raise RuntimeError(f"unexpected public resource option {typ}")
                    legal.append(option);ids.append(a)
                if len(legal)>1:
                    deck=[(int(c[0])-1,int(c[1])) for c in rows(o,"deck") if c[0]]
                    inv=[9+c for c,u in deck]+[8+sizes["cards"]+int(r[0]) for r in rows(o,"relics") if r[0]]
                    base=[*o[:5],*o[12:15],len(deck),sum(u>0 for c,u in deck),sum(c==sizes["strikes"] for c,u in deck),
                          sum(c==sizes["defends"] for c,u in deck),sum(o[17:22]>0),o[16]]
                    examples.append((o.copy(),mask.copy(),ids,[seed,0,kind,base,inv,legal,0]))
            if not game.step(action):raise RuntimeError("baseline resource path diverged")
    engine.set_resource_policy(*cfg)
    data=pack([v[3] for v in examples],len(cfg[2]))
    with torch.inference_mode(): predictions=net(*data[:-1]).numpy()
    maximum=0.;counts=Counter();by_screen=Counter();worst={}
    for sample,z in zip(examples,predictions):
        o,mask,ids,row=sample
        q=np.asarray(engine.scores(w,o,mask))[ids]
        z=z[:len(ids)]
        err=float(np.max(np.abs((q-q.max())/8-(z-z.max()))))
        if err>by_screen[row[2]]:
            worst[row[2]]=(row[0],row[3],row[5],(q-q.max()).tolist(),(8*(z-z.max())).tolist())
        by_screen[row[2]]=max(by_screen[row[2]],err)
        maximum=max(maximum,err)
        counts[row[2]]+=1
    if maximum>1e-4:
        print(dict(by_screen),str(worst)[:4500])
    if len(counts)!=4 or maximum>1e-4:raise RuntimeError(f"resource export mismatch {counts}, {maximum}")
    result=dict(menus_by_type=dict(counts),max_logit_error=maximum)
    write(out/"parity.json",result)
    print(result)


if __name__=="__main__":
    main()
