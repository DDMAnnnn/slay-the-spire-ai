"""Try card offers and upgrades in paired public practice fights, then measure earned runs."""
import argparse
import json
from pathlib import Path

import heart_native as engine
import numpy as np

from .deck_training import funnel
from .heart_curriculum import read_model,save_model
from .public_belief import evaluate
from .replay import apply_model,heart_encounters
from .train import statistics,write


def panel_chunks(out,name,w,start,count,threads):
    rows_path=out/f"{start}-{name}.npy"
    rows=np.load(rows_path) if rows_path.is_file() else np.empty((0,6),dtype=int)
    for offset in range(len(rows),count,1024):
        batch=engine.batch(w,start+offset,min(1024,count-offset),threads)
        rows=np.concatenate([rows,np.asarray(batch)])
        tmp=out/f"{name}.tmp.npy"
        np.save(tmp,rows);tmp.replace(rows_path)
    return {**statistics(rows),**funnel(rows),"heart_encounter_seeds":heart_encounters(w,start,rows)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=Path("build/inventory-search"))
    p.add_argument("--threads",type=int,default=8)
    args=p.parse_args()
    args.out.mkdir(exist_ok=True,parents=True)
    parent=read_model("build/trajectory-teacher/challenger.json.gz")
    w=apply_model("first",parent)
    report=dict(screen_start=193000000,validation=[],test={})
    for settings in ((0,0,0.),(3,0,35.),(3,1,100.),(6,1,75.),(6,1,220.),(9,0,130.)):
        engine.set_inventory_search(*settings)
        result=evaluate(w,193000000,1024,args.threads)
        result["settings"]=settings
        report["validation"].append(result)
        write(args.out/"screen.json",report)
    def rank(v):
        return v["first_boss"]+2*v["second_boss"]+.1*v["act3"]+4*v["act4"]
    settings=max(report["validation"][1:],key=rank)["settings"]
    report["inventory"]=settings
    belief=json.loads(Path("build/public-belief/report.json").read_text())["selected"]
    report["belief"]=belief
    engine.set_inventory_search(0,0,0.)
    engine.collect_beliefs(w,belief["seed"],belief["count"])
    for name,use_belief,use_inventory in (("parent",False,False),("inventory",False,True),
                                          ("combined",True,True),("belief",True,False)):
        engine.set_belief(belief["draws"] if use_belief else 0,belief["scope"],belief["scale"])
        engine.set_inventory_search(*(settings if use_inventory else (0,0,0.)))
        report["test"][name]=panel_chunks(args.out,name,w,194000000,32768,args.threads)
        print(name,report["test"][name],flush=True)
        write(args.out/"report.json",report)
    save_model(args.out/"combined.json.gz",{**parent,"public_belief":belief,"inventory_search":settings})


if __name__=="__main__":
    main()
