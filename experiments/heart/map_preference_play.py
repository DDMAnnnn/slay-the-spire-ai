"""Select on fresh ordinary starts and evaluate the frozen router on 32,768 others."""
from pathlib import Path
import json

import heart_native as engine
import numpy as np

from .heart_curriculum import read_model
from .replay import apply_model
from .route_rollouts import summarize
from .train import write


def main():
    out = Path("build/map-preferences")
    path = out / "play.json"
    report = json.loads(path.read_text()) if path.exists() else {"selection": {}, "test": {}}
    parent = Path("build/route-irl/selected-whole_map.json.gz")
    def run(name, start, count, phase):
        if name in report[phase]:
            return
        w = apply_model("first", read_model(parent if name == "parent" else out / f"{name}.json.gz"))
        parts = []
        for offset in range(0, count, 1024):
            part = out / f"{phase}-{name}-{offset}.npy"
            if not part.exists():
                ids = list(range(start + offset, start + offset + min(1024, count - offset)))
                np.save(part, np.asarray(engine.batch_seeds(w, ids, 3)))
            parts.append(np.load(part))
        results = np.concatenate(parts)
        hits = (start + np.flatnonzero(results[:, 6])).tolist()
        report[phase][name] = dict(**summarize(results), heart_encounter_seeds=hits,
                                   victory_seeds=(start + np.flatnonzero(results[:, 4])).tolist())
        write(path, report)
        print(phase, name, summarize(results), flush=True)
    for name in ("parent", "heart", "milestones"):
        run(name, 252000000, 8192, "selection")
    def rank(name):
        v = report["selection"][name]
        return v["encounters"], v["act4"], v["second"], v["boss"]
    pick = max(("heart", "milestones"), key=rank)
    for name in ("parent", pick):
        run(name, 253000000, 32768, "test")
    a = set(report["test"][pick]["heart_encounter_seeds"])
    b = set(report["test"]["parent"]["heart_encounter_seeds"])
    report.update(selected=pick, gained=len(a-b), lost=len(b-a), shared=len(a&b))
    write(path, report)


if __name__ == "__main__":
    main()
