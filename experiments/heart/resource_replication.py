"""Replicate the upgrade-only policy and test adding the campfire head on a new ordinary panel."""
from pathlib import Path

import heart_native as engine
import numpy as np

from .heart_curriculum import read_model, save_model
from .replay import apply_model
from .train import write


def main():
    out=Path("build/resource-replication")
    out.mkdir(parents=True,exist_ok=True)
    parent=read_model("build/stratified-gate/challenger.json.gz")
    trained=read_model("build/resource-policy/challenger.json.gz")
    start=231000000
    ids=np.arange(start,start+32768)
    result={"start":start,"arms":{}}
    controls=None
    for name,mix in (("parent",[0.]*4),("upgrade",[0.,0.,.7,0.]),("camp_upgrade",[.7,0.,.7,0.])):
        model=parent if name=="parent" else {**parent,"resource_policy":[*trained["resource_policy"][:-1],mix]}
        w=apply_model("first",model)
        f=out/f"{name}.npy"
        if not f.exists():
            np.save(f,np.concatenate([np.asarray(engine.batch_seeds(w,chunk.tolist(),4)) for chunk in np.array_split(ids,32)]))
        data=np.load(f)
        if controls is None:controls=data
        hit=data[:,6]>0;old=controls[:,6]>0
        result["arms"][name]=dict(runs=len(ids),act3=int((data[:,1]>=3).sum()),act4=int((data[:,1]>=4).sum()),
            first_boss=int((data[:,0]>=50).sum()),second_boss=int((data[:,0]>=51).sum()),wins=int(data[:,4].sum()),
            encounters=int(hit.sum()),gained=int((hit&~old).sum()),lost=int((old&~hit).sum()),
            heart_encounter_seeds=ids[hit].tolist(),victory_seeds=ids[data[:,4]>0].tolist(),truncated=int(data[:,5].sum()))
        save_model(out/f"{name}.json.gz",model)
        write(out/"report.json",result)
        print(name,result["arms"][name],flush=True)


if __name__=="__main__":
    main()
