"""Fit card play biases and a state-dependent combat potion policy from paired rollouts."""
import argparse
import gzip
import json
from pathlib import Path

import heart_native as engine
import numpy as np
import sts_env as sts

from .boss_relics import compare
from .replay import heart_encounters, load_model, set_combat_preferences
from .train import statistics, write


def collect(weights, start, count, threads):
    drinks, cards = [], []
    for offset in range(0, count, 256):
        a, b = engine.preference_practice(weights, start + offset, min(256, count-offset), threads)
        drinks.extend(a)
        cards.extend(b)
    return np.asarray(drinks), np.asarray(cards)


def fit_item_values(data, item_count, shrinkage=32):
    ids = data[:, 0].astype(int)
    x, y = data[:, 1:-1], data[:, -1]
    # Pool evidence for rare items. Each ID still learns its own intercept and state effects.
    penalty = shrinkage * np.eye(x.shape[1])
    shared = np.linalg.solve(x.T @ x + penalty, x.T @ y)
    table = []
    for i in range(item_count):
        local = x[ids == i]
        table.append(np.linalg.solve(local.T @ local + penalty, local.T @ y[ids == i] + penalty @ shared))
    return np.asarray(table)


def fit_cards(data):
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(2)
    prior = torch.tensor(engine.initial_play_base(), dtype=torch.float32)
    base = prior.clone().requires_grad_()
    opt = torch.optim.Adam([base], lr=.5)
    # Skip equal-return pairs. End turn (-1) has a fixed zero coefficient.
    rows = torch.tensor(data[data[:, -1] != 0], dtype=torch.float32)
    a, b = rows[:, 0].long(), rows[:, 1].long()
    for _ in range(100):
        delta = base - prior
        margin = rows[:, 2] + rows[:, 3]*delta[a] - rows[:, 4]*delta[b]
        loss = (F.softplus(-rows[:, 5].sign()*margin/10) * rows[:, 5].abs().clamp(max=15)).mean()
        loss = loss + .00025 * delta.square().sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return base.detach().tolist()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prior", type=Path, default=Path("build/first-boss-relics/challenger.json.gz"))
    p.add_argument("--out", type=Path, default=Path("build/combat-preferences"))
    p.add_argument("--threads", type=int, default=10)
    args = p.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    weights = load_model("first", args.prior)
    samples = {}
    for name, start, n in (("train", 60000000, 8192), ("validation", 61000000, 1024)):
        drinks, cards = collect(weights, start, n, args.threads)
        samples[name] = drinks, cards
        print(name, len(drinks), len(cards), flush=True)
    drinks, cards = samples["train"]
    table = fit_item_values(drinks, len(sts.POTION_NAMES))
    bases = fit_cards(cards)
    checks = samples["validation"][0]
    predictions = (checks[:, 1:-1] * table[checks[:, 0].astype(int)]).sum(axis=1)
    means = np.bincount(drinks[:, 0].astype(int), weights=drinks[:, -1], minlength=len(table))
    means /= np.maximum(1, np.bincount(drinks[:, 0].astype(int), minlength=len(table)))
    report = dict(train_examples=[len(drinks), len(cards)], validation_examples=list(map(len,samples["validation"])),
                  potion_rmse=float(np.sqrt(np.mean((predictions-checks[:, -1])**2))),
                  potion_id_only_rmse=float(np.sqrt(np.mean((means[checks[:, 0].astype(int)]-checks[:, -1])**2))),
                  validation_start=62000000, test_start=63000000, validation={}, test={})
    def panel(pref, start, n):
        set_combat_preferences(pref)
        return np.asarray(engine.batch(weights, start, n, args.threads))

    def drink_policy(margin):
        return dict(drink_table=table.tolist(), margin=margin)

    margin_trials = {}
    for margin in (0., .5, 1., 2., 4.):
        summary = statistics(panel(drink_policy(margin), 62000000, 4096))
        margin_trials[margin] = summary
    report["validation"] = margin_trials
    margin = max(margin_trials, key=lambda m: (margin_trials[m]["act3"], margin_trials[m]["mean_floor"]))
    report["selected_margin"] = margin
    versions = dict(previous={}, cards=dict(play_base=bases), drinks=drink_policy(margin),
                    combined=dict(**drink_policy(margin), play_base=bases))
    with gzip.open(args.prior, "rt", encoding="utf8") as f:
        model = json.load(f)
    for name, filename in (("combined", "challenger.json.gz"), ("cards", "cards-only.json.gz"),
                           ("drinks", "drinks-only.json.gz")):
        model["combat_preferences"] = versions[name]
        with gzip.open(args.out/filename, "wt", encoding="utf8") as f:
            json.dump(model, f)
    for name, pref in versions.items():
        result = panel(pref, 63000000, 32768)
        if name == "previous":
            baseline = result
        summary = dict(**statistics(result), **compare(baseline, result),
                       heart_encounter_seeds=heart_encounters(weights, 63000000, result))
        report["test"][name] = summary
        print(name, summary, flush=True)
    write(args.out/"report.json", report)


if __name__ == "__main__":
    main()
