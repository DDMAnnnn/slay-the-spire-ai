"""Fit draft and event preferences from matched alternatives on training runs."""
import argparse
import json
from pathlib import Path

import heart_native as engine
import numpy as np
from .train import rollouts, statistics, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("prior", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--episodes", type=int, default=16384)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--seed", type=int, default=7000000)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    policy = json.loads(args.prior.read_text())
    events = json.loads((args.prior.parent / "event-table.json").read_text())
    engine.set_event_table(events)
    write(args.out / "event-table.json", events)
    w = np.array(policy["weights"])
    sums = np.zeros((3, engine.feature_count))
    counts = np.zeros_like(sums)
    for offset in range(0, args.episodes, 256):
        n = min(256, args.episodes-offset)
        rows = np.array(engine.practice(w.tolist(), args.seed+offset, n, args.threads))
        if rows.size:
            idx = (rows[:, 0].astype(int), rows[:, 1].astype(int))
            np.add.at(sums, idx, rows[:, 2])
            np.add.at(counts, idx, 1)
        if offset % 2048 == 0:
            print(offset+n, int(counts.sum()), "paired observations", flush=True)
    # Give the pooled estimate the weight of 40 observations in each act.
    # An option seen only twice in Act 3 then borrows most of its estimate from the other acts.
    common = sums.sum(axis=0)/(counts.sum(axis=0)+30)
    effects = (sums + 40*common)/(counts+40)
    best = -np.inf
    trials = []
    for scale in (0., .75, 1.5, 3., 6.):
        table = (effects*scale).tolist()
        engine.set_choice_table(table)
        val = rollouts(w, 6800000, 4096, args.threads)
        stats = statistics(val)
        score = (val[:,0]/56 + .5*(val[:,1]>=2) + 5*(val[:,1]>=3) + 30*(val[:,1]==4) + 100*val[:,4]).mean()
        trials.append(dict(scale=scale, score=float(score), **stats))
        if score > best:
            best = score
            write(args.out/"choice-table.json", table)
        print(trials[-1], flush=True)
    policy["choice_learning"] = dict(episodes=args.episodes, examples=int(counts.sum()), validation=trials)
    write(args.out/"policy.json", policy)


if __name__ == "__main__":
    main()
