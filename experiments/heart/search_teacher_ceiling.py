"""Measure the teacher directly to separate search quality from distillation loss."""
import json
from pathlib import Path
import subprocess

import torch

from .expert_state_learning import play, send, statistics
from .train import write
from .transfer_probe import EXAMPLE, model_code


def main():
    torch.set_num_threads(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net, _ = model_code().load(EXAMPLE / "weights.pt", device)
    net.load_state_dict(torch.load("build/public-expert-intent/imitation.pt", map_location=device, weights_only=True))
    out = Path("build/teacher-ceiling")
    out.mkdir(exist_ok=True, parents=True)
    path = out / "report.json"
    result = json.loads(path.read_text()) if path.exists() else {}
    cmd = [str(Path("build/public-expert-traces/python313/python.exe").resolve()),
           str(Path(__file__).with_name("expert_state_worker.py"))]
    with subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="ascii") as worker:
        try:
            result.update(json.loads(worker.stdout.readline()))
            for split in ("validation", "test"):
                group = result.setdefault(split, {})
                ref = json.loads(Path(f"build/public-root-compatible/{split}-parent.json").read_text())
                group["student"] = statistics(ref)
                for budget in (64, 384, 2048, 8192):
                    if str(budget) in group:
                        continue
                    rows = play(worker, net, device, split, search_budget=budget)
                    if [(r["seed"], r["index"]) for r in rows] != [(r["seed"], r["index"]) for r in ref]:
                        raise ValueError("different initial fights")
                    metrics = statistics(rows)
                    metrics["paired"] = dict(gained=sum(a["win"] and not b["win"] for a,b in zip(rows,ref)),
                                             lost=sum(b["win"] and not a["win"] for a,b in zip(rows,ref)))
                    group[str(budget)] = metrics
                    write(out / f"{split}-{budget}.json", rows)
                    write(path, result)
                    print(split, budget, metrics["all"], metrics["paired"], flush=True)
        finally:
            if worker.poll() is None:
                send(worker.stdin, {"stop": True})
                worker.stdin.close()


if __name__ == "__main__":
    main()
