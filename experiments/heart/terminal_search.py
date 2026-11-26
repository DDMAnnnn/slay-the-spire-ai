"""Compare full-fight search against the current rollout policy on matched earned battles."""
from pathlib import Path
from time import perf_counter
import json

import heart_native as engine
import numpy as np

from .heart_curriculum import read_model, save_model
from .matched_fights import summary
from .replay import apply_model
from .train import write


VARIANTS = {"rollout": (0, 0.), "mean96": (96, 0.), "mixed96": (96, .35),
            "mean384": (384, 0.), "mixed384": (384, .35)}


def main():
    out = Path("build/terminal-search")
    out.mkdir(exist_ok=True, parents=True)
    path = out / "report.json"
    report = json.loads(path.read_text()) if path.exists() else {}
    parent = read_model("build/route-irl/selected-whole_map.json.gz")
    template = read_model("build/stacked-search/joint.json.gz")["public_belief"]
    cfg = {**template, "stasis": True, "draws": 3, "scope": 1, "scale": 8.,
           "enrich": 192, "plans": 1, "win_value": 32., "action_limit": 7}
    w = apply_model("first", {**parent, "public_belief": cfg})

    def enable(name):
        engine.set_belief(0 if name == "base" else cfg["draws"], cfg["scope"], cfg["scale"])
        engine.set_terminal_search(*VARIANTS.get(name, (0, 0.)))

    def panel(phase, start, count, names):
        rows = {k: [] for k in names}
        seconds = dict.fromkeys(names, 0.)
        for offset in range(0, count, 8):
            pending = [n for n in names if not (out / f"{phase}-{offset}-{n}.npz").exists()]
            if pending:
                enable("base")
                bank = engine.MatchedFights(w, start + offset, min(8, count-offset), 3)
                for name in pending:
                    enable(name)
                    begin = perf_counter()
                    data = np.column_stack((bank.info, bank.evaluate(w, 3)))
                    np.savez(out / f"{phase}-{offset}-{name}.npz", rows=data, seconds=perf_counter()-begin)
            for name in names:
                with np.load(out / f"{phase}-{offset}-{name}.npz") as result:
                    rows[name].append(result["rows"])
                    seconds[name] += float(result["seconds"])
        stats = {}
        baseline = np.concatenate(rows["base"])
        for name in names:
            data = np.concatenate(rows[name])
            if not np.array_equal(data[:, :6], baseline[:, :6]):
                raise ValueError("different starting fights")
            if name == "base" and not np.array_equal(data[:, 1], data[:, 6]):
                raise ValueError("control did not reproduce")
            stats[name] = dict(**summary(data), seconds=seconds[name])
        report[phase] = stats
        write(path, report)
        print(phase, {k: (v["fatal"]["wins"],v["control"]["wins"]) for k,v in stats.items()}, flush=True)

    panel("selection", 254000000, 48, ("base", *VARIANTS))
    def rank(name):
        v = report["selection"][name]
        return (v["fatal"]["wins"] + v["control"]["wins"],
                v["fatal"]["mean_exit_hp"] + v["control"]["mean_exit_hp"])
    selected = max((k for k in VARIANTS if k != "rollout"), key=rank)
    save_model(out / "challenger.json.gz", {**parent, "public_belief": {**cfg, "terminal_search": VARIANTS[selected]}})
    report["selected"] = selected
    panel("heldout", 255000000, 160, ("base", "rollout", selected))


if __name__ == "__main__":
    main()
