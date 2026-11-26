"""Let the fitted public value function rerank two plausible search moves."""
import argparse
import json
from pathlib import Path
import subprocess

import torch

from .expert_state_learning import play, send, statistics
from .train import write
from .transfer_probe import EXAMPLE, model_code


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correction",type=Path,help="Frozen action-effect predictor fitted on paired trials")
    args=parser.parse_args()
    frozen=Path("build/afterstate-critic/critic.json")
    out=args.correction.parent if args.correction else frozen.parent
    if not frozen.exists():
        raise FileNotFoundError(frozen)
    torch.set_num_threads(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net, _ = model_code().load(EXAMPLE / "weights.pt", device)
    net.load_state_dict(torch.load("build/public-expert-intent/imitation.pt", map_location=device, weights_only=True))
    exe = Path("build/public-expert-traces/python313/python.exe").resolve()
    cmd = [str(exe),str(Path(__file__).with_name("expert_state_worker.py"))]
    path = out / "play.json"
    report = json.loads(path.read_text()) if path.exists() else {"validation": {}, "test": {}}
    with subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="ascii") as worker:
        try:
            bank = json.loads(worker.stdout.readline())
            report["banks"] = bank["banks"]
            def run(split, weight):
                name = str(weight)
                if name in report[split]:
                    return
                basefile = Path("build/shared-search-screen") / f"{split}-384-2.json"
                control = json.loads(basefile.read_text())
                if weight:
                    config = dict(model=str(frozen),weight=weight,worlds=3)
                    if args.correction:config['correction']=str(args.correction)
                    rows = play(worker,net,device,split,search_budget=384,search_mode=2,afterstate=config)
                else:
                    rows = control
                keys = lambda table: [(r["seed"], r["index"]) for r in table]
                if keys(rows) != keys(control):
                    raise ValueError("counterfactuals started at different fights")
                summary = statistics(rows)["all"]
                summary.update(changed=sum(r.get("critic_changed",0) for r in rows),
                               compared=sum(r.get("critic_compared",0) for r in rows),
                               gained=sum(a["win"] and not b["win"] for a,b in zip(rows,control)),
                               lost=sum(b["win"] and not a["win"] for a,b in zip(rows,control)))
                report[split][name] = summary
                write(out / f"{split}-{name}.json",rows)
                write(path,report)
                print(split,name,summary,flush=True)
            for weight in ((0.,1.) if args.correction else (0.,.5,1.5)):
                run("validation",weight)
            def rank(weight):
                v=report["validation"][str(weight)]
                return v["wins"],v["exit_hp"]
            chosen=1. if args.correction else max((.5,1.5),key=rank)
            report["selected_weight"]=chosen
            for weight in (0.,chosen):
                run("test",weight)
            write(path,report)
        finally:
            if worker.poll() is None:
                try:
                    send(worker.stdin,{"stop":True})
                    worker.stdin.close()
                except (OSError,BrokenPipeError):
                    pass


if __name__ == "__main__":
    main()
