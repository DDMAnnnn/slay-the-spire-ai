"""Test learned postcombat utility inside search on earned fatal fights and winning controls."""
import json
from pathlib import Path

import heart_native as engine
import numpy as np

from .heart_curriculum import read_model
from .matched_fights import summary
from .replay import apply_model
from .train import write


def main():
    out=Path("build/resource-effects");dst=out/"search.json"
    r=json.loads(dst.read_text()) if dst.exists() else {}
    parent=read_model("build/route-irl/selected-whole_map.json.gz")
    sample=read_model("build/stacked-search/joint.json.gz")["public_belief"]
    cfg={**sample,"stasis":True,"draws":3,"scope":1,"scale":8.,"enrich":192,
         "plans":1,"win_value":32.,"action_limit":7}
    w=apply_model("first",{**parent,"public_belief":cfg})
    table=json.loads((out/"tables.json").read_text())
    typical=json.loads((out/"fit.json").read_text())["typical_potion_cost"]
    choices={"parent":(0.,0.),"half":(0.,.5),"full":(0.,1.),"joint":(.5,1.)}
    def measure(phase,start,n,names):
        if phase in r:return
        chunks={name:[] for name in names}
        for off in range(0,n,24):
            engine.reset_resource_effects()
            engine.set_belief(0,cfg["scope"],cfg["scale"])
            bank=engine.MatchedFights(w,start+off,min(24,n-off),3)
            info=np.asarray(bank.info);info[:,0]+=off
            engine.set_belief(cfg["draws"],cfg["scope"],cfg["scale"])
            for name in names:
                gain,mix=choices[name]
                engine.set_resource_effects(table["hp"],table["potion"],gain,mix,typical)
                chunks[name].append(np.column_stack((info,np.asarray(bank.evaluate(w,3)))))
        r[phase]={}
        for name,parts in chunks.items():
            a=np.concatenate(parts)
            control=np.concatenate(chunks["parent"])
            if not np.array_equal(a[:,:6],control[:,:6]):raise ValueError("mismatched fights")
            s=summary(a)
            s.update(gained=int(((a[:,6]>0)&(control[:,6]==0)).sum()),lost=int(((a[:,6]==0)&(control[:,6]>0)).sum()))
            r[phase][name]=s
        write(dst,r);print(phase,r[phase],flush=True)
    measure("selection",271000000,72,list(choices))
    def rank(name):
        a=r["selection"][name]
        return a["control"]["wins"]+a["fatal"]["wins"],a["control"]["mean_exit_hp"]+a["fatal"]["mean_exit_hp"]
    best=max(("half","full","joint"),key=rank);r["selected"]=best
    measure("test",272000000,160,["parent",best]);write(dst,r)


if __name__=="__main__":
    main()
