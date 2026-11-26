"""Replicate the learned map and combine it with the best independently frozen search/draft player."""
import argparse
import json
from pathlib import Path

import heart_native as engine
import numpy as np

from .heart_curriculum import read_model,save_model
from .inventory_search import panel_chunks
from .replay import apply_model
from .train import write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("arm",choices=("replicate","joint"));p.add_argument("--threads",type=int,default=4)
    args=p.parse_args();out=Path("build/route-search-transfer")/args.arm;out.mkdir(parents=True,exist_ok=True)
    cfg=read_model("build/route-irl/selected-whole_map.json.gz")["map_policy"]
    if args.arm=="joint":
        start=194000000
        base=read_model("build/stacked-search/joint.json.gz")
        result={"parent":json.loads(Path("build/stacked-search/report.json").read_text())[str(start)]["joint"]}
    else:
        start=248000000;base=read_model("build/stratified-gate/challenger.json.gz");result={}
    file=out/"report.json"
    if file.exists():result=json.loads(file.read_text())
    for name,m in (("parent",base),("map",{**base,"map_policy":cfg})):
        if name in result:continue
        w=apply_model("first",m);save_model(out/f"{name}.json.gz",m)
        result[name]=panel_chunks(out,name,w,start,32768,args.threads)
        write(file,result);print(args.arm,name,result[name],flush=True)
    a=set(result["map"]["heart_encounter_seeds"]);b=set(result["parent"]["heart_encounter_seeds"])
    result.update(start=start,gained=len(a-b),lost=len(b-a),shared=len(a&b));write(file,result)


if __name__=="__main__":
    main()
