"""Learn which tactical search to run when the fight begins, using paired outcomes."""
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from .train import write


ARMS = ((384, 2), (2048, 2), (384, -1), (2048, -1))


def features(act, floor, entry_hp, maximum):
    health = entry_hp / max(1, maximum)
    boss = float(floor in (16, 33, 50, 51, 56))
    stage = max(0, floor - (0, 17, 34, 52)[act - 1]) / 16
    return [1., health, health ** 2, stage, boss, boss * health,
            *[float(act == a) for a in (2, 3, 4)]]


def paired_data(folder, split, entries):
    arms = [json.loads((folder / f"{split}-{budget}-{mode}.json").read_text())
            for budget, mode in ARMS]
    keys = [(row["seed"], row["index"]) for row in arms[0]]
    for arm in arms[1:]:
        if keys != [(r["seed"], r["index"]) for r in arm]:
            raise ValueError("tactical outcomes refer to different beginnings")
    x = []
    for row, key in zip(arms[0], keys):
        f = entries[key]
        x.append(features(f["act"], f["floor"], f["entry_hp"], row["max_hp"]))
    hp = np.array([[r["hp"] / r["max_hp"] for r in arm] for arm in arms]).T
    win = np.array([[r["win"] for r in arm] for arm in arms]).T
    cost = np.array([[r["labels"] * budget for r in arm] for arm, (budget, _) in zip(arms, ARMS)]).T
    folds = [int.from_bytes(hashlib.sha256(seed.encode()).digest()[:4], "big") % 5 for seed, _ in keys]
    return np.array(x), win, hp, cost, np.array(folds)


def fit(x, y, penalty):
    # Only differences from the same fight's control teach the search choice.
    delta = y - y[:, :1]
    reg = np.eye(x.shape[1]) * penalty
    return np.linalg.solve(x.T @ x + reg, x.T @ delta)


def report(choice, win, hp, cost):
    i = np.arange(len(choice))
    freq = np.bincount(choice, minlength=len(ARMS)) / len(choice)
    return dict(fights=len(i), wins=int(win[i, choice].sum()),
                mean_exit_health_fraction=float(hp[i, choice].mean()),
                simulations=int(cost[i, choice].sum()),
                gained=int((win[i, choice] & ~win[:, 0]).sum()),
                lost=int((~win[i, choice] & win[:, 0]).sum()),
                counts=np.bincount(choice, minlength=len(ARMS)).tolist(),
                random_same_mix_expected_wins=float((win.sum(0) * freq).sum()),
                random_same_mix_expected_simulations=float((cost.sum(0) * freq).sum()))


def main():
    root = Path("build/public-expert-traces")
    folder = Path("build/shared-search-screen")
    dest = Path("build/search-allocation")
    dest.mkdir(parents=True, exist_ok=True)
    with gzip.open(root / "replays.jsonl.gz", "rt", encoding="utf8") as src:
        entries = {(str(row["seed"]), i): f for row in map(json.loads, src)
                   for i, f in enumerate(row["fights"])}
    x, wins, hp, cost, folds = paired_data(folder, "validation", entries)
    # Higher exit HP breaks near ties. A survival always outweighs the whole HP bonus.
    y = wins + .25 * hp
    heldout = {}
    for strength in (1., 8., 64.):
        decisions = np.zeros(len(x), dtype=int)
        for fold in np.unique(folds):
            tr = folds != fold
            decisions[~tr] = (x[~tr] @ fit(x[tr], y[tr], strength)).argmax(1)
        heldout[strength] = report(decisions, wins, hp, cost)
    def rank(s):
        v = heldout[s]
        return v["wins"] + .25 * len(x) * v["mean_exit_health_fraction"], -v["simulations"]
    strength = max(heldout, key=rank)
    weights = fit(x, y, strength)
    test, wins, hp, cost, _ = paired_data(folder, "test", entries)
    choices = (test @ weights).argmax(1)
    result = dict(arms=ARMS, development_fights=len(x), grouped_cv=heldout, ridge=strength,
                  test=report(choices, wins, hp, cost),
                  fixed=[report(np.full(len(test), i), wins, hp, cost) for i in range(len(ARMS))])
    write(dest / "report.json", result)
    write(dest / "policy.json", dict(arms=ARMS, weights=weights.tolist()))
    print(result["test"], flush=True)


if __name__ == "__main__":
    main()
