"""Deploy the inferred deck-plan preferences and compare them with a single residual expert."""
import json
from pathlib import Path

import heart_native as engine
import numpy as np
import sts_env as sts
import torch

from .heart_curriculum import read_model,save_model
from .human_menus import STARTERS,TOKENS
from .latent_draft import Plans
from .replay import apply_model,rows
from .route_rollouts import summarize
from .train import write


def audit(parent,model,groups,out):
    cfg=model["draft_plans"];net=Plans(groups);net.load_state_dict(torch.load(out/f"plan-{groups}.pt",weights_only=True));net.eval()
    torch.set_num_threads(1)
    c,k,b,u,r=map(np.asarray,json.loads(Path("build/human-strategy/fitted.json").read_text())["context"]["config"])
    w=apply_model("first",model);samples=[]
    for seed in range(275100000,275100048):
        e=sts.Env();e.reset(seed,20,final_act=True,character="ironclad")
        while not e.done:
            o,m=e.observe(),e.mask();q=np.asarray(engine.scores(w,o,m))
            ix=np.flatnonzero(m[81:145]);opt=rows(o,"choices");kind=set(opt[ix,0])
            if not o[7] and len(ix)>1 and ((o[6]==2 and 1 in kind and 6 in kind and kind<={1,6,14}) or (o[6]==3 and kind=={2,6})):
                samples.append((o.copy(),m.copy(),q.copy(),ix))
            if not e.step(int(q.argmax())):raise ValueError("illegal draft")
        if len(samples)>=110:break
    apply_model("first",parent);error=0.;changed=0
    card_names=sts.CARD_IDS
    relic_names=Path("build/rl-fork/include/constants/Relics.h").read_text(encoding="utf8")
    import re
    relic_ids=re.findall(r'"([^"]*)"',relic_names.split("relicIds[]",1)[1].split("};",1)[0])
    with torch.inference_mode():
        for o,m,q,ix in samples:
            base=np.asarray(engine.scores(w,o,m));ctx=np.zeros(TOKENS);boss=int(o[6])==3
            for v in rows(o,"deck"):
                if v[0] and card_names[int(v[0])-1] not in STARTERS:
                    t=int(v[0])+1;ctx[t]=min(3,ctx[t]+1)
            for v in rows(o,"relics"):
                if v[0] and relic_ids[int(v[0])-1]!="Burning Blood":ctx[1+engine.card_count+int(v[0])]=1
            features=rows(o,"choices")[ix];ids=[];ups=[]
            for v in features:
                typ=int(v[0]);ids.append(0 if typ==6 else 1 if typ==14 else 1+int(v[1])+(engine.card_count if boss else 0))
                ups.append(bool(v[2]) and not boss and typ==1)
            act=int(o[4])-1;act=max(0,min(2,act));h=np.tanh(ctx@c/np.sqrt(max(1,ctx.sum())))
            prior=b[act,ids]+u[ids]*ups+r[ids]*np.log1p(ctx[ids])+k[ids]@h
            z=net(torch.tensor(ctx[None],dtype=torch.float32),torch.tensor([act]),torch.tensor([ids]),
                  torch.tensor(prior[None],dtype=torch.float32))[0][0].numpy()
            delta=z-prior;delta-=delta[ids.index(0)]
            error=max(error,float(np.max(np.abs(cfg["scales"][boss]*delta-(q[ix+81]-base[ix+81])))))
            changed+=int(q.argmax()!=base.argmax())
    if error>.001 or not len(samples):raise ValueError(("draft plan parity",len(samples),error))
    return dict(menus=len(samples),changed=changed,max_error=error)


def main():
    out=Path("build/latent-draft");fit=json.loads((out/"fit.json").read_text());chosen=int(fit["selected"])
    parent=read_model("build/human-macro-replication/fire.json.gz")
    params={g:json.loads((out/f"plan-{g}.json").read_text()) for g in (chosen,1)}
    plans={"parent":(0,[]),"one":(1,[12.,18.]),"light":(chosen,[4.,6.]),"medium":(chosen,[12.,18.]),
           "strong":(chosen,[30.,40.]),"cards":(chosen,[12.,0.])}
    def candidate(name):
        n,s=plans[name]
        return {**parent,"draft_plans":{**params[n],"scales":s}} if n else parent
    path=out/"play.json";r=json.loads(path.read_text()) if path.exists() else {"selection":{},"test":{}}
    if "audit" not in r:r["audit"]=audit(parent,candidate("medium"),chosen,out);write(path,r)
    def run(phase,name):
        if name in r[phase]:return
        panel=267771 if phase=="selection" else 275771
        ids=np.random.default_rng(panel).bit_generator.random_raw(32768)
        w=apply_model("first",candidate(name));parts=[]
        for i in range(0,len(ids),1024):
            f=out/f"{phase}-{name}-{i}.npy"
            if phase=="selection" and name=="parent":
                f=Path("build/resource-effects")/f"selection-parent-{i}.npy"
                if not f.exists():raise ValueError("missing same-seed control")
            elif not f.exists():np.save(f,np.asarray(engine.batch_seeds(w,ids[i:i+1024].tolist(),3)))
            parts.append(np.load(f))
        a=np.concatenate(parts);hits=set(map(str,ids[a[:,6]>0]))
        v=dict(**summarize(a),panel=panel,heart_encounter_seeds=sorted(hits),victory_seeds=list(map(str,ids[a[:,4]>0])))
        if name!="parent":
            old=set(r[phase]["parent"]["heart_encounter_seeds"]);v.update(gained=len(hits-old),lost=len(old-hits))
        r[phase][name]=v;write(path,r);print(phase,name,summarize(a),flush=True)
    for name in plans:run("selection",name)
    def rank(name):
        v=r["selection"][name];return v["encounters"],v["wins"],v["act4"]
    best=max(("light","medium","strong","cards"),key=rank)
    r["selected"]=best;plans["matched_one"]=(1,plans[best][1])
    for name in ("parent",best,"matched_one"):run("test",name)
    write(path,r);save_model(out/"selected.json.gz",candidate(best))


if __name__=="__main__":main()
