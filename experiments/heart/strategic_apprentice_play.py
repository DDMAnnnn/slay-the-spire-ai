"""Check deployment math and test the shared strategic actor on ordinary starts."""
import json
from pathlib import Path

import heart_native as engine
import numpy as np
import sts_env as sts
import torch

from .heart_curriculum import read_model, save_model
from .replay import apply_model
from .route_rollouts import summarize
from .strategic_apprentice import CandidatePolicy, represent
from .train import write


def check(model,w,out):
    net=CandidatePolicy();net.load_state_dict(torch.load(out/"chosen.pt",map_location="cpu",weights_only=True));net.eval()
    params=model["strategic_apprentice"]
    helpers={};worst=0.;torch.set_num_threads(1);samples=[]
    for seed in range(262420000,262420048):
        e=sts.Env();e.reset(seed,20,final_act=True,character="ironclad")
        while not e.done:
            o,m=e.observe(),e.mask();native=np.asarray(engine.scores(w,o,m))
            if not o[7] and o[6]!=5 and int(o[9])!=32 and m[81:145].sum()>1:
                samples.append((o.copy(),m.copy(),native.copy()))
            if not e.step(int(native.argmax())):raise ValueError("illegal strategic move")
        if len(samples)>=800:break
    engine.reset_strategic_apprentice()
    for o,m,native in samples:
        base=np.asarray(engine.scores(w,o,m))
        enc=represent(o,list(np.flatnonzero(m)),base)
        if enc is None:continue
        typ,bag,items=enc
        keys=list(items);tok,category,features=zip(*items.values())
        with torch.no_grad():
            z=net(torch.tensor([typ]),torch.tensor([bag]),torch.tensor([tok]),torch.tensor([category]),
                  torch.tensor(np.asarray([features]),dtype=torch.float32),torch.ones((1,len(keys)),dtype=torch.bool))[0].numpy()
        diff=np.max(np.abs((native[keys]-native[keys].max())-8*(z-z.max())))
        worst=max(worst,float(diff));helpers[typ]=helpers.get(typ,0)+1
    engine.set_strategic_apprentice(*params)
    if worst>.0005 or len(helpers)<6:raise ValueError((worst,helpers))
    return dict(menus=helpers,max_error=worst)


def main():
    out=Path("build/strategic-apprentice");path=out/"play.json"
    r=json.loads(path.read_text()) if path.exists() else {"selection":{},"test":{}}
    base=read_model("build/route-irl/selected-whole_map.json.gz");model=read_model(out/"apprentice.json.gz")
    if "audit" not in r:
        r["audit"]=check(model,apply_model("first",model),out);write(path,r)
    cfgs={"parent":[0.]*10,"light":[.30]*10,"strong":[.70]*10,"replace":[1.]*10,
          "inventory":[0.,0.,.7,.7,.7,0.,.7,.7,.7,0.],"services":[.7,.7,0.,0.,.7,.7,.7,.7,.7,0.]}
    def run(phase,name,n):
        if name in r[phase]:return
        ids=np.random.default_rng(262012 if phase=="selection" else 263012).bit_generator.random_raw(n)
        m={**base,"strategic_apprentice":[*model["strategic_apprentice"][:-1],cfgs[name]]}
        w=apply_model("first",m);parts=[]
        for i in range(0,n,1024):
            p=out/f"{phase}-{name}-{i}.npy"
            if not p.exists():np.save(p,np.asarray(engine.batch_seeds(w,ids[i:i+1024].tolist(),3)))
            parts.append(np.load(p))
        a=np.concatenate(parts);hits=set(map(str,ids[a[:,6]>0]))
        r[phase][name]=dict(**summarize(a),heart_encounter_seeds=sorted(hits),victory_seeds=list(map(str,ids[a[:,4]>0])))
        if name!="parent":
            old=set(r[phase]["parent"]["heart_encounter_seeds"]);r[phase][name].update(gained=len(hits-old),lost=len(old-hits))
        if phase=="test":save_model(out/f"selected-{name}.json.gz",m)
        write(path,r);print(phase,name,summarize(a),flush=True)
    for name in cfgs:run("selection",name,8192)
    def rank(name):
        v=r["selection"][name];return v["encounters"],v["act4"],v["second"],v["boss"]
    selected=max((n for n in cfgs if n!="parent"),key=rank)
    r["selected"]=selected
    for name in ("parent",selected):run("test",name,32768)
    write(path,r)


if __name__=="__main__":main()
