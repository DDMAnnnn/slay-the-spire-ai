"""Repeat the frozen human draft plus public search; isolate reward search on the original panel."""
import argparse
import json
from pathlib import Path

from .heart_curriculum import read_model, save_model
from .inventory_search import panel_chunks
from .replay import apply_model
from .train import write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("arm",choices=("replication","combat_only","search_control"))
    p.add_argument("--threads",type=int,default=4)
    args=p.parse_args()
    out=Path("build/joint-search-replication")/args.arm
    out.mkdir(parents=True,exist_ok=True)
    m=read_model("build/stacked-search/joint.json.gz")
    start=216000000
    if args.arm=="combat_only":
        start=194000000
        m.pop("inventory_search",None)
    if args.arm=="search_control":
        m.pop("historical_menu")
    save_model(out/"model.json.gz",m)
    w=apply_model("first",m)
    result=panel_chunks(out,args.arm,w,start,32768,args.threads)
    write(out/"report.json",dict(start=start,arm=args.arm,result=result))
    print(args.arm,result,flush=True)


if __name__=="__main__":
    main()
