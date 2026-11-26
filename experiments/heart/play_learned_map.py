"""Check map-score parity and compare learned routes with identical combat and resource policies."""
import json
from pathlib import Path

import heart_native as engine
import numpy as np
import sts_env as sts
import torch

from .heart_curriculum import read_model, save_model
from .replay import apply_model, rows
from .route_irl import RoutePolicy, features
from .train import write


def audit(out,name,m):
    w=apply_model("first",m);net=RoutePolicy(name=="whole_map").eval()
    net.load_state_dict(torch.load(out/f"{name}.pt",map_location="cpu",weights_only=True))
    torch.set_num_threads(1)
    count=0;worst=0.
    for seed in range(242200000,242200032):
        e=sts.Env();e.reset(seed,20,final_act=True,character="ironclad")
        while not e.done:
            o,mask=e.observe(),e.mask()
            q=np.asarray(engine.scores(w,o,mask))
            legal=mask[sts.ACT_CHOOSE:sts.ACT_CHOOSE+7].astype(bool)
            if not o[7] and o[6]==5 and 0<=o[11]+1<15 and legal.sum()>1:
                board=rows(o,"map").reshape(15,7,2)
                rooms=np.where(board[:,:,0],board[:,:,0]-1,8).astype(np.int64)
                key=np.zeros((15,7),np.int64)
                if not o[13] and 0<=o[24]<7 and 0<=o[25]<15:key[int(o[25]),int(o[24])]=1
                ctx=[int(o[4]),*o[:3],float((o[17:22]>0).sum()),o[22],*o[12:15]]
                y=int(o[11])+1
                args=[torch.as_tensor(np.array(x)[None,...]) for x in
                        (rooms,board[:,:,1],key,y,features(ctx,y),legal)]
                with torch.inference_mode():z=net(*args)[0].numpy()[legal]
                actual=q[sts.ACT_CHOOSE:sts.ACT_CHOOSE+7][legal]
                worst=max(worst,float(np.max(np.abs((actual-actual.max())/8-(z-z.max())))))
                count+=1
            if not e.step(int(q.argmax())):raise RuntimeError("illegal map audit action")
    if not count or worst>1e-4:raise RuntimeError(f"map score mismatch on {count} choices, worst {worst}")
    return dict(branches=count,max_logit_error=worst)


def main():
    out=Path("build/route-irl");file=out/"play.json"
    r=json.loads(file.read_text()) if file.exists() else {"audit":{},"screen":{},"test":{}}
    parent=read_model("build/stratified-gate/challenger.json.gz")
    models={n:read_model(out/f"{n}.json.gz") for n in ("next_room","whole_map")}
    def mixture(n,t):
        m=models[n]
        return {**m,"map_policy":[*m["map_policy"][:-1],t]}
    def run(start,n,m):
        w=apply_model("first",m);ids=np.arange(start,start+n)
        v=np.concatenate([np.asarray(engine.batch_seeds(w,part.tolist(),4)) for part in np.array_split(ids,max(1,n//1024))])
        return dict(n=n,act3=int((v[:,1]>=3).sum()),first_boss=int((v[:,0]>=50).sum()),second_boss=int((v[:,0]>=51).sum()),
                   act4=int((v[:,1]>=4).sum()),heart_encounter_seeds=ids[v[:,6]>0].tolist(),
                   victory_seeds=ids[v[:,4]>0].tolist(),truncated=int(v[:,5].sum()))
    for name,m in models.items():
        if name not in r["audit"]:r["audit"][name]=audit(out,name,m);write(file,r)
    r["screen_start"]=242000000;r["test_start"]=244000000
    if "parent" not in r["screen"]:
        r["screen"]["parent"]=run(242000000,4096,parent);write(file,r)
    for name in models:
        s=r["screen"].setdefault(name,{})
        for t in (.35,.7,1.):
            if str(t) not in s:s[str(t)]=run(242000000,4096,mixture(name,t));write(file,r)
        def rank(t):
            v=s[t];return (len(v["heart_encounter_seeds"]),v["act4"],v["second_boss"],v["first_boss"],v["act3"])
        r.setdefault("selected",{})[name]=float(max(s,key=rank))
    for name in ("parent",*models):
        if name in r["test"]:continue
        m=parent if name=="parent" else mixture(name,r["selected"][name])
        if name!="parent":save_model(out/f"selected-{name}.json.gz",m)
        r["test"][name]=run(244000000,32768,m);write(file,r)
        print(name,r["test"][name],flush=True)


if __name__=="__main__":
    main()
