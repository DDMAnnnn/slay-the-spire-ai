"""Compare snapshots of one outcome-trained actor on the same full-game starts."""
import argparse
from pathlib import Path

import numpy as np
import torch

from .direct_actor import fit
from .heart_curriculum import read_model
from .recovery import panel
from .train import write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=Path("build/actor-epochs"))
    parser.add_argument("--threads",type=int,default=8)
    args=parser.parse_args()
    args.out.mkdir(exist_ok=True,parents=True)
    source=Path("build/direct-actor")
    torch.set_num_threads(3)
    data=np.concatenate([np.load(source/f"{name}.npy") for name in ("reference","staged")])
    valid=np.load(source/"valid.npy")
    original=read_model(source/"outcomes.json.gz")
    report=dict(start=112000000,full_validation_runs=4096,training_rows=len(data),epochs={})
    def snapshot(epoch,actor,gain,agreement):
        if epoch not in (1,4,8,13,16,32):
            return
        weights=actor.export()
        if epoch==13:
            errors=[float(np.max(np.abs(np.asarray(old)-np.asarray(new))))
                    for old,new in zip(original["direct_actor"],weights)]
            report["epoch13_max_weight_difference_from_archived"]=max(errors)
        summary,_=panel({**original,"direct_actor":weights},112000000,4096,args.threads)
        summary.update(agreement=agreement,single_move_gain=gain)
        report["epochs"][epoch]=summary
        print("epoch",epoch,"move gain",round(gain,3),"Act 3",summary["act3"],
              "Heart",len(summary["heart_encounter_seeds"]),flush=True)
        write(args.out/"report.json",report)
    fit(data,valid,.25,epochs=32,on_epoch=snapshot)


if __name__=="__main__":
    main()
