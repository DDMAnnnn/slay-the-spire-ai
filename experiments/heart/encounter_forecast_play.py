"""Verify the encounter model across the Python/C++ boundary and select it on ordinary starts."""
import json
from pathlib import Path

import heart_native as engine
import numpy as np
import sts_env as sts
import torch

from .encounter_forecast import Forecast, context, tokens
from .heart_curriculum import read_model,save_model
from .replay import apply_model, rows
from .route_rollouts import summarize
from .train import write


def audit(base,config):
    torch.set_num_threads(1)
    net=Forecast(len(config["weights"][1]),True)
    net.load_state_dict(torch.load("build/encounter-forecast/inventory.pt",map_location="cpu",weights_only=True));net.eval()
    w=apply_model("first",{**base,"encounter_forecast":config})
    samples=[]
    for seed in range(269100000,269100060):
        e=sts.Env();e.reset(seed,20,final_act=True,character="ironclad")
        while not e.done:
            o,m=e.observe(),e.mask();q=np.asarray(engine.scores(w,o,m))
            screen=int(o[6]);legal=np.flatnonzero(m[81:145])
            menu=rows(o,"choices")
            types=set(menu[legal,0].astype(int))
            eligible=(screen==2 and types=={1,6}) or screen==3 or (screen==4 and o[26] in (3,4))
            if not o[7] and o[4]<=3 and eligible and len(legal)>1:
                samples.append((o.copy(),m.copy(),q.copy()))
            if not e.step(int(q.argmax())):raise ValueError("illegal forecast action")
        if len(samples)>=100:break
    engine.reset_encounter_forecast()
    changed=0;worst=0.;checks=0
    with torch.inference_mode():
        for o,m,q in samples:
            prior=np.asarray(engine.scores(w,o,m));changed+=int(q.argmax()!=prior.argmax())
            row=dict(act=int(o[4]),floor=int(o[3]),hp=float(o[0]),max_hp=float(o[1]),
                     deck=[(int(c[0])-1,int(c[1]>0)) for c in rows(o,"deck") if c[0]],
                     relics=[int(c[0])-1 for c in rows(o,"relics") if c[0]],
                     potions=[int(v)-1 for v in o[17:22] if v])
            bank=config["encounters"][str(row["act"])]
            bank=[(i,(1.-config["boss"])*p) for i,p in bank]+[(int(o[15])-1,config["boss"])]
            enemies=torch.tensor([v[0] for v in bank]);mass=torch.tensor([v[1] for v in bank])
            def utility(entry):
                items=torch.tensor([tokens(entry)]).expand(len(bank),-1)
                x=torch.tensor([context(entry)]).expand(len(bank),-1)
                p=net(items,enemies,x).sigmoid()
                return float((mass*p[:,0]*(.55+.45*p[:,1])).sum())
            initial=utility(row)
            for i,c in enumerate(rows(o,"choices")):
                if not m[81+i]:continue
                edit={**row,"deck":row["deck"][:],"relics":row["relics"][:]}
                if c[0]==1:
                    card=(int(c[1])-1,int(c[2]>0))
                    if o[6]==4:
                        if o[26]==3 and card[1]:continue
                        edit["deck"].remove(card)
                        if o[26]==3:edit["deck"].append((card[0],1))
                    else:edit["deck"].append(card)
                elif int(o[6])==3 and c[0]==2:edit["relics"].append(int(c[1])-1)
                else:continue
                pred=config["gain"]*(utility(edit)-initial)
                worst=max(worst,abs(pred-(q[81+i]-prior[81+i])));checks+=1
    if worst>.0008 or not checks:raise ValueError(("forecast score parity",worst,checks))
    return dict(menus=len(samples),options=checks,max_error=worst,changed=changed)


def main():
    out=Path("build/encounter-forecast")
    base=read_model("build/human-macro-replication/fire.json.gz")
    forecast=json.loads((out/"weights.json").read_text())
    resources=Path("build/resource-effects")
    table=json.loads((resources/"tables.json").read_text())
    typical=json.loads((resources/"fit.json").read_text())["typical_potion_cost"]
    settings={"parent":(0.,0.,0.),"future":(180.,.30,0.),"strong":(550.,.30,0.),
              "boss":(400.,.75,0.),"joint":(400.,.50,.50),"potion":(0.,0.,.50)}
    def candidate(name):
        gain,boss,drink=settings[name];m={**base}
        if gain:m["encounter_forecast"]={**forecast,"gain":gain,"boss":boss}
        if drink:m["resource_effects"]={**table,"gain":drink,"mix":0.,"typical":typical}
        return m
    dest=out/"play.json";report=json.loads(dest.read_text()) if dest.exists() else {"selection":{},"test":{}}
    if "audit" not in report:
        report["audit"]=audit(base,candidate("future")["encounter_forecast"]);write(dest,report)
    def run(split,name):
        if name in report[split]:return
        panel=267771 if split=="selection" else 269771
        ids=np.random.default_rng(panel).bit_generator.random_raw(32768)
        w=apply_model("first",candidate(name));chunks=[]
        for offset in range(0,len(ids),1024):
            f=out/f"{split}-{name}-{offset}.npy"
            if split=="selection" and name in ("parent","potion"):
                f=resources/f"selection-{'parent' if name=='parent' else 'half'}-{offset}.npy"
                if not f.exists():raise ValueError("missing same-seed control")
            elif not f.exists():np.save(f,np.asarray(engine.batch_seeds(w,ids[offset:offset+1024].tolist(),3)))
            chunks.append(np.load(f))
        a=np.concatenate(chunks);hits=set(map(str,ids[a[:,6]>0]))
        r=dict(**summarize(a),panel=panel,heart_encounter_seeds=sorted(hits),victory_seeds=list(map(str,ids[a[:,4]>0])))
        if name!="parent":
            before=set(report[split]["parent"]["heart_encounter_seeds"]);r.update(gained=len(hits-before),lost=len(before-hits))
        report[split][name]=r;write(dest,report);print(split,name,summarize(a),flush=True)
    for name in settings:run("selection",name)
    def rank(name):
        v=report["selection"][name];return v["encounters"],v["wins"],v["act4"]
    best=max(("future","strong","boss","joint"),key=rank);report["selected"]=best
    for name in ("parent","potion",best):run("test",name)
    if best=="joint":
        # Remove just the potion correction without changing the selected encounter mixture.
        settings["same_forecast"]=(400.,.50,0.);run("test","same_forecast")
    write(dest,report);save_model(out/"selected.json.gz",candidate(best))


if __name__=="__main__":
    main()
