"""Compare openings under the current human-trained draft on ordinary A20 games."""
from pathlib import Path

import heart_native as engine
import numpy as np
import sts_env as sts

from .heart_curriculum import read_model, save_model
from .replay import apply_model, rows
from .train import write


def main():
    out=Path("build/starter-relic-transfer")
    out.mkdir(parents=True,exist_ok=True)
    parent=read_model("build/stratified-gate/challenger.json.gz")
    original=np.asarray(read_model("build/learned-opening/challenger.json.gz")["neow"])
    env=sts.Env();swap=set()
    for seed in range(237100000,237100128):
        env.reset(seed,20,final_act=True,character="ironclad")
        opts=rows(env.observe(),"choices")[:4]
        if any(int(o[0])!=sts.CHOICE_KINDS.index("NEOW_OPTION") for o in opts):raise ValueError("incomplete Neow menu")
        swap.add(int(opts[3,1])-1)
    if len(swap)!=1:raise ValueError("starter exchange has inconsistent ID")
    preserve=original.copy()
    preserve[swap.pop(),:]=original.min()-100.
    ids=np.arange(237000000,237032768)
    report=dict(start=int(ids[0]),arms={})
    for name,table in (("parent",None),("all_offers",original),("retain",preserve)):
        m=parent if table is None else {**parent,"neow":table.tolist()}
        w=apply_model("first",m)
        f=out/f"{name}.npy"
        if not f.exists():
            np.save(f,np.concatenate([np.asarray(engine.batch_seeds(w,s.tolist(),4)) for s in np.array_split(ids,16)]))
        data=np.load(f);hits=data[:,6]>0
        if name=="parent":old=hits
        report["arms"][name]=dict(n=len(ids),act2=int((data[:,1]>=2).sum()),act3=int((data[:,1]>=3).sum()),
            act4=int((data[:,1]>=4).sum()),heart_encounter_seeds=ids[hits].tolist(),victory_seeds=ids[data[:,4]>0].tolist(),
            gained=int((hits&~old).sum()),lost=int((old&~hits).sum()),truncated=int(data[:,5].sum()))
        save_model(out/f"{name}.json.gz",m);write(out/"report.json",report)
        print(name,report["arms"][name],flush=True)


if __name__=="__main__":
    main()
