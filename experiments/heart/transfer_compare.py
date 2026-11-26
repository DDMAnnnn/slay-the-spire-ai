"""Compare the frozen source, imitation, and PPO on fresh identical Ironclad starts."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from .heart_curriculum import read_model
from .recovery import panel
from .train import write
from .transfer_probe import EXAMPLE,evaluate,model_code


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training",type=Path,default=Path("build/ironclad-transfer"))
    p.add_argument("--start",type=int,default=158000000)
    p.add_argument("--count",type=int,default=1024)
    p.add_argument("--batch",type=int,default=256)
    p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    args=p.parse_args()
    torch.set_num_threads(3)
    torch.distributions.Distribution.set_default_validate_args(False)
    report=json.loads((args.training/"report.json").read_text())
    stages={row["update"]:row["stage"] for row in report["updates"]}
    def rank(row):
        return len(row["heart_encounter_seeds"]),row["act4"],row["act3"],row["act2"]
    snapshots={}
    for stage in ("cloning","ppo"):
        candidates=[row for row in report["checkpoints"] if stages[row["update"]]==stage]
        snapshots[stage]=max(candidates,key=rank)
    src=model_code()
    actor,_=src.load(EXAMPLE/"weights.pt",args.device)
    parent=read_model("build/trajectory-teacher/challenger.json.gz")
    ref,_=panel(parent,args.start,args.count,6)
    output=dict(start=args.start,selection="Heart, Act 4, Act 3, Act 2 on validation starts",parent=ref,
                updates={s:c["update"] for s,c in snapshots.items()},results={})
    for name,info in {"frozen":None,**snapshots}.items():
        if info:
            path=args.training/info["path"]
            ck=torch.load(path,map_location=args.device,weights_only=True)
            actor.load_state_dict(ck["agent"])
        else:
            path=EXAMPLE/"weights.pt"
        result=evaluate(actor,src.no_unpick,args.device,"all",parent,args.start,args.count,args.batch,6,"ironclad")
        result["checkpoint_sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
        output["results"][name]=result
        print(name,result,flush=True)
        write(args.training/"comparison.json",output)


if __name__=="__main__":
    main()
