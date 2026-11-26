"""Replay ordinary starts and measure where a policy loses HP or dies in Act 2."""
import argparse
from collections import defaultdict
from pathlib import Path

import heart_native as engine
import numpy as np
import sts_env as sts

from .replay import load_model
from .train import write


def diagnose(checkpoint,start,n):
    w=load_model("first",checkpoint)
    encounters=defaultdict(list)
    rooms=defaultdict(list)
    arrivals=bosses=0
    for seed in range(start,start+n):
        env=sts.Env()
        env.reset(seed,20,final_act=True,character="ironclad")
        for action in engine.trace(w,seed):
            if not env.step(action):
                raise RuntimeError(f"divergent replay on seed {seed}")
        episode=env.episode()
        arrivals+=episode["act"]>=2
        for room in episode["path"]:
            if room["act"]!=2 or not room["fought"]:
                continue
            bosses+=room["room"]=="BOSS"
            row=(room["hp_before"],room["hp_after"],room["hp_after"]<=0 and episode["lost"]
                 and room["floor"]==episode["floor"],room["max_hp"])
            encounters[room["encounter"]].append(row)
            rooms[room["room"]].append(row)
    def group(items):
        result={}
        for name,rows in items.items():
            data=np.array(rows)
            result[name]=dict(entries=len(data),deaths=int(data[:,2].sum()),
                              median_entry_hp=float(np.median(data[:,0])),
                              median_entry_fraction=round(float(np.median(data[:,0]/data[:,3])),3),
                              mean_net_hp_lost=round(float((data[:,0]-data[:,1]).mean()),2))
        return dict(sorted(result.items(),key=lambda pair:-pair[1]["deaths"]))
    return dict(start=start,runs=n,act2=arrivals,act2_boss_entries=bosses,
                room=group(rooms),encounter=group(encounters))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint",type=Path)
    parser.add_argument("--start",type=int,required=True)
    parser.add_argument("--runs",type=int,default=4096)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    report=diagnose(args.checkpoint,args.start,args.runs)
    write(args.out,report)
    print(report["act2"],report["room"],flush=True)
