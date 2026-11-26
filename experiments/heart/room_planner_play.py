"""Test survival planning with candidate routes scored from learned room transitions."""
from functools import lru_cache
import json
from pathlib import Path

import heart_native as engine
import numpy as np
import sts_env as sts

from .heart_curriculum import read_model,save_model
from .replay import apply_model
from .route_rollouts import summarize
from .train import write


def audit(parent,model):
    cfg=model["room_planner"]
    t=np.array(cfg["transition"]).reshape(3,4,2,9,9,10)
    tail=np.array(cfg["value"]).reshape(3,4,2,2,9)
    w=apply_model("first",model);samples=[]
    for seed in range(273100000,273100048):
        e=sts.Env();e.reset(seed,20,final_act=True,character="ironclad")
        while not e.done:
            o,m=e.observe(),e.mask();q=np.asarray(engine.scores(w,o,m))
            if not o[7] and o[6]==5 and o[11]<14 and m[81:88].sum()>=2:
                samples.append((o.copy(),m.copy(),q.copy()))
            if not e.step(int(q.argmax())):raise ValueError("illegal map choice")
        if len(samples)>=80:break
    w=apply_model("first",parent);worst=0.;changed=0
    for o,m,q in samples:
        baseline=np.asarray(engine.scores(w,o,m))
        start=int(o[11])+1;last=min(14,start+cfg["horizon"]-1)
        act=int(o[4])-1;pot=int(any(o[17:22]));green=int(bool(o[13]))
        off,_,_=sts.OBS_LAYOUT["map"];grid=o[off:off+210].reshape(15,7,2).astype(int)
        @lru_cache(None)
        def solve(row,x,g):
            raw,edges=grid[row,x];tile=raw-1
            if tile==3 and (x,row)==(o[24],o[25]):tile=8;after_green=1
            else:after_green=g
            if row==last:
                later=tail[act,min(3,(last+1)//4),pot,after_green]
            else:
                opts=[solve(row+1,x+d,after_green) for d in (-1,0,1) if 0<=x+d<7 and edges&(1<<(d+1))]
                later=np.max(opts,axis=0) if opts else np.zeros(9)
            return t[act,min(3,row//4),pot,tile,:,1:]@later
        moves=np.flatnonzero(m[81:88]);vals=[]
        for x in moves:
            z=solve(start,x,green);vals.append(np.interp(8*o[0]/max(1,o[1]),np.arange(9),z))
        expected=cfg["gain"]*(np.array(vals)-vals[np.argmax(baseline[81+moves])])
        worst=max(worst,float(np.max(np.abs(q[81+moves]-baseline[81+moves]-expected))))
        changed+=int(q.argmax()!=baseline.argmax())
    if worst>.0005 or not changed:raise ValueError(("route backup mismatch",len(samples),changed,worst))
    return dict(maps=len(samples),changed=changed,max_error=worst)


def main():
    out=Path("build/room-transitions");data=json.loads((out/"tables.json").read_text())
    parent=read_model("build/human-macro-replication/fire.json.gz")
    configs={"parent":None,"short":(3,7,100.),"deep":(12,7,150.),"last_two":(9,6,250.),
             "act2":(7,2,300.),"act3":(12,4,350.)}
    def model(name):
        if name=="parent":return parent
        horizon,acts,gain=configs[name]
        return {**parent,"room_planner":{**data,"horizon":horizon,"acts":acts,"gain":gain}}
    path=out/"play.json";r=json.loads(path.read_text()) if path.exists() else {"selection":{},"test":{}}
    if "audit" not in r:r["audit"]=audit(parent,model("deep"));write(path,r)
    def run(phase,name):
        if name in r[phase]:return
        panel=267771 if phase=="selection" else 273771
        ids=np.random.default_rng(panel).bit_generator.random_raw(32768);w=apply_model("first",model(name))
        parts=[]
        for off in range(0,len(ids),1024):
            f=out/f"{phase}-{name}-{off}.npy"
            if phase=="selection" and name=="parent":
                f=Path("build/resource-effects")/f"selection-parent-{off}.npy"
                if not f.exists():raise ValueError("missing same-seed baseline")
            elif not f.exists():np.save(f,np.asarray(engine.batch_seeds(w,ids[off:off+1024].tolist(),4)))
            parts.append(np.load(f))
        a=np.concatenate(parts);hits=set(map(str,ids[a[:,6]>0]))
        stat=dict(**summarize(a),panel=panel,heart_encounter_seeds=sorted(hits),victory_seeds=list(map(str,ids[a[:,4]>0])))
        if name!="parent":
            baseline=set(r[phase]["parent"]["heart_encounter_seeds"]);stat.update(gained=len(hits-baseline),lost=len(baseline-hits))
        r[phase][name]=stat;write(path,r);print(phase,name,summarize(a),flush=True)
    for k in configs:run("selection",k)
    def rank(k):
        v=r["selection"][k];return v["encounters"],v["wins"],v["act4"],v["second"]
    best=max((n for n in configs if n!="parent"),key=rank);r["selected"]=best
    for k in ("parent",best):run("test",k)
    write(path,r);save_model(out/"selected.json.gz",model(best))


if __name__=="__main__":main()
