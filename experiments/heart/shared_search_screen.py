"""Compare independent-world and shared public combat trees under matched simulation budgets."""
import json
from pathlib import Path
import subprocess
import torch

from .expert_state_learning import play, send, statistics
from .transfer_probe import EXAMPLE, model_code
from .train import write


def main():
    out=Path("build/shared-search-screen");out.mkdir(parents=True,exist_ok=True)
    path=out/"report.json";r=json.loads(path.read_text()) if path.exists() else {"validation":{},"test":{}}
    torch.set_num_threads(2);device="cuda" if torch.cuda.is_available() else "cpu"
    net,_=model_code().load(EXAMPLE/"weights.pt",device)
    net.load_state_dict(torch.load("build/public-expert-intent/imitation.pt",map_location=device,weights_only=True))
    cmd=[str(Path("build/public-expert-traces/python313/python.exe").resolve()),str(Path(__file__).with_name("expert_state_worker.py"))]
    with subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,encoding="ascii") as worker:
        try:
            r.update(json.loads(worker.stdout.readline()))
            def qualify(split,budget,mode):
                key=f"{budget}-{mode}"
                if key in r[split]:return
                if mode==-1:
                    fights=json.loads(Path(f"build/teacher-ceiling/{split}-{budget}.json").read_text())
                else:
                    fights=play(worker,net,device,split,search_budget=budget,search_mode=mode)
                control=json.loads(Path(f"build/teacher-ceiling/{split}-{budget}.json").read_text())
                assert [(v["seed"],v["index"]) for v in fights]==[(v["seed"],v["index"]) for v in control]
                stats=statistics(fights)
                stats["paired"]=dict(gained=sum(a["win"] and not b["win"] for a,b in zip(fights,control)),
                                      lost=sum(b["win"] and not a["win"] for a,b in zip(fights,control)))
                r[split][key]=stats;write(out/f"{split}-{key}.json",fights);write(path,r)
                print(split,key,stats["all"],stats["paired"],flush=True)
            for mode in (-1,0,2,3):qualify("validation",384,mode)
            def rank(mode):
                a=r["validation"][f"384-{mode}"]["all"];return a["wins"],a["exit_hp"]
            selected=max((0,2,3),key=rank);r["selected"]=selected
            for split in ("validation","test"):
                for budget in ((2048,) if split=="validation" else (384,2048)):
                    for mode in (-1,selected):qualify(split,budget,mode)
            write(path,r)
        finally:
            if worker.poll() is None:
                try:send(worker.stdin,{"stop":True});worker.stdin.close()
                except (BrokenPipeError,OSError):pass


if __name__=="__main__":main()
