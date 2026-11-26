"""Separate the search changes and test frozen policies after naturally earned late-game prefixes."""
import argparse
from pathlib import Path

import heart_native as engine
import numpy as np

from .heart_curriculum import read_model
from .matched_fights import SETTINGS, configure, summary
from .replay import apply_model
from .train import write


def isolate(out, w, threads):
    root = SETTINGS["current_search"]
    variants = {}
    # Each option changes one property of the root search; the tactical portfolio
    # normally searches nine actions, so clamp that factor separately here.
    for name, index, value in (("draws", 0, 8), ("hallways", 1, 2), ("scale", 2, 12.),
                               ("tactics", 3, 3), ("survival_bonus", 4, 90.), ("enemy_bank", 5, True)):
        setting = list(root)
        setting[index] = value
        variants[name] = (setting, 5)
    variants["candidate_count"] = (root, 9)
    reference = np.load("build/matched-fights/202000000-current_search.npy")
    results = {}
    for name, (setting, action_limit) in variants.items():
        path = out / f"{name}.npy"
        if not path.exists():
            output = []
            for offset in range(0, 256, 32):
                configure(SETTINGS["parent"])
                bank = engine.MatchedFights(w, 202000000 + offset, 32, threads)
                configure(setting)
                engine.set_belief_action_limit(action_limit)
                meta = np.asarray(bank.info).copy()
                meta[:, 0] += offset
                output.append(np.column_stack((meta, bank.evaluate(w, threads))))
            np.save(path, np.concatenate(output))
        rows = np.load(path)
        if not np.array_equal(rows[:, :6], reference[:, :6]):
            raise ValueError("matched controls or encounter identities differ")
        paired = {}
        for group, won in (("fatal", 0), ("control", 1)):
            selected = rows[:, 1] == won
            a, b = reference[selected, 6] > 0, rows[selected, 6] > 0
            paired[group] = dict(gains=int((~a & b).sum()), losses=int((a & ~b).sum()))
        results[name] = dict(settings=setting, action_limit=action_limit, outcome=summary(rows), paired=paired)
        write(out / "isolated.json", results)
    engine.set_belief_action_limit(0)


def late_prefixes(out, w, threads):
    scout_seed = 207000000
    scout = out / "scout.npy"
    configure(SETTINGS["parent"])
    if not scout.exists():
        np.save(scout, np.asarray(engine.batch(w, scout_seed, 32768, threads)))
    floors = np.load(scout)[:, 0]
    rng = np.random.default_rng(20709)
    report = {"scout_start": scout_seed, "scout_count": len(floors), "floors": {}}
    for floor in (49, 32):
        # Qualify on arrival only. The random draw has no knowledge of later outcomes.
        possible = np.flatnonzero(floors >= floor)
        seeds = scout_seed + np.sort(rng.choice(possible, size=min(96, len(possible)), replace=False))
        group = report["floors"][floor] = {"eligible": len(possible), "seeds": seeds.tolist(), "results": {}}
        for name in ("parent", "current_search", "deeper_portfolio"):
            path = out / f"floor-{floor}-{name}.npy"
            rows = np.load(path).tolist() if path.exists() else []
            for i in range(len(rows), len(seeds), 12):
                configure(SETTINGS["parent"])
                bank = engine.Bank(w, seeds[i:i+12].tolist(), floor, threads)
                configure(SETTINGS[name])
                rows.extend(bank.progress(w, threads))
                tmp = path.with_suffix(".tmp.npy")
                np.save(tmp, rows)
                tmp.replace(path)
            x = np.asarray(rows)
            group["results"][name] = dict(count=len(x), act3=int((x[:, 1] >= 3).sum()),
                second_boss_floor=int((x[:, 0] >= 51).sum()), act4=int((x[:, 1] == 4).sum()),
                heart_encounter_seeds=seeds[x[:, 2] > 0].tolist(), wins=int(x[:, 3].sum()),
                truncated=int(x[:, 4].sum()))
            write(out / "prefixes.json", report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("build/search-components"))
    p.add_argument("--threads", type=int, default=3)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    parent = read_model("build/trajectory-teacher/challenger.json.gz")
    w = apply_model("first", parent)
    engine.collect_beliefs(w, 190000000, 3072)
    engine.enrich_late_particles(w, 192)
    isolate(args.out, w, args.threads)
    late_prefixes(args.out, w, args.threads)


if __name__ == "__main__":
    main()
