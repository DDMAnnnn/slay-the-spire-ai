"""Select the learned opportunity-cost correction and measure the frozen policy on new starts."""
import json
from pathlib import Path

import heart_native as engine
import numpy as np

from .heart_curriculum import read_model, save_model
from .replay import apply_model
from .route_rollouts import summarize
from .train import write


def main():
    out = Path("build/resource-effects")
    tables = json.loads((out / "tables.json").read_text())
    fit = json.loads((out / "fit.json").read_text())
    parent = read_model("build/human-macro-replication/fire.json.gz")
    path = out / "play.json"
    report = json.loads(path.read_text()) if path.exists() else {"selection": {}, "test": {}}
    gains = {"parent": 0., "half": .5, "medium": 1.5, "strong": 4.}
    def with_gain(name):
        if name == "parent":
            return parent
        config = dict(**tables, typical=fit["typical_potion_cost"], mix=0., gain=gains[name])
        return {**parent, "resource_effects": config}
    def run(phase, name):
        if name in report[phase]:
            return
        panel = 267771 if phase == "selection" else 268771
        ids = np.random.default_rng(panel).bit_generator.random_raw(32768)
        model = with_gain(name)
        w = apply_model("first", model)
        chunks = []
        for start in range(0, len(ids), 1024):
            part = out / f"{phase}-{name}-{start}.npy"
            if not part.exists():
                np.save(part, np.asarray(engine.batch_seeds(w, ids[start:start+1024].tolist(), 4)))
            chunks.append(np.load(part))
        a = np.concatenate(chunks)
        reached = set(map(str, ids[a[:, 6] > 0]))
        result = dict(**summarize(a), panel=panel, heart_encounter_seeds=sorted(reached),
                      victory_seeds=list(map(str, ids[a[:, 4] > 0])))
        if name != "parent":
            baseline = set(report[phase]["parent"]["heart_encounter_seeds"])
            result.update(gained=len(reached-baseline), lost=len(baseline-reached))
        report[phase][name] = result
        write(path, report)
        print(phase, name, summarize(a), flush=True)
    for name in gains:
        run("selection",name)
    def rank(name):
        s = report["selection"][name]
        return s["encounters"], s["wins"], s["act4"], s["second"]
    best = max((k for k in gains if k != "parent"), key=rank)
    report["selected"] = best
    run("test","parent")
    run("test",best)
    write(path,report)
    save_model(out / "selected.json.gz",with_gain(best))


if __name__ == "__main__":
    main()
