"""Test a two-decision public search on matched fights and ordinary A20 starts."""
import argparse
import json
from pathlib import Path

import heart_native as engine
import numpy as np

from .heart_curriculum import read_model, save_model
from .inventory_search import panel_chunks
from .matched_fights import SETTINGS, configure, summary
from .public_belief import evaluate
from .replay import apply_model
from .train import write


def select(name):
    configure(SETTINGS["current_search"] if name == "three_draws" else SETTINGS["survival"])
    engine.set_continuation(4 if name in ("both", "lookahead") else 0, name in ("both", "free_plays"))


def matched(w, start, count, variants, threads):
    samples = {k: [] for k in variants}
    for i in range(0, count, 16):
        configure(SETTINGS["parent"])
        engine.set_continuation(0, False)
        bank = engine.MatchedFights(w, start + i, min(16, count-i), threads)
        for name in variants:
            select(name)
            samples[name].append(np.column_stack((bank.info, bank.evaluate(w, threads))))
    return {name: summary(np.concatenate(rows)) for name, rows in samples.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--threads", type=int, default=3)
    p.add_argument("--out", type=Path, default=Path("build/chain-search"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    location = args.out / "report.json"
    report = json.loads(location.read_text()) if location.exists() else {}
    parent = read_model("build/trajectory-teacher/challenger.json.gz")
    w = apply_model("first", parent)
    engine.collect_beliefs(w, 190000000, 3072)
    engine.enrich_late_particles(w, 192)
    select("both")
    if "audit" not in report:
        report["audit"] = dict(zip(("decisions", "shared_groups", "heldout_worlds", "different_first_moves"),
                                   engine.continuation_audit(w, 209000000, 16)))
        write(location, report)
    if "matched" not in report:
        report["matched"] = matched(w, 209100000, 128, ("three_draws", "six_draws", "both"), args.threads)
        write(location, report)
    # Both groups are sampled from the same parent trajectories. Losing one
    # old win offsets a rescue in the balanced diagnostic.
    def total(scores):
        return scores["fatal"]["wins"] + scores["control"]["wins"]
    matched_scores = report["matched"]
    if total(matched_scores["both"]) > total(matched_scores["six_draws"]) and "isolated" not in report:
        report["isolated"] = matched(w, 209100000, 128, ("lookahead", "free_plays"), args.threads)
        write(location, report)
    screen = report.setdefault("screen", {})
    for name in ("three_draws", "six_draws", "both"):
        if name not in screen:
            select(name)
            screen[name] = evaluate(w, 210000000, 1024, args.threads)
            write(location, report)
    def rank(scores):
        return scores["first_boss"] + 2*scores["second_boss"] + .1*scores["act3"] + 4*scores["act4"]
    if rank(screen["both"]) <= max(rank(screen["six_draws"]), rank(screen["three_draws"])):
        report["qualification"] = "Joint change did not beat both comparators on the ordinary screen"
        write(location, report)
        return
    select("both")
    cfg = dict(seed=190000000, count=3072, draws=6, scope=2, scale=8., plans=1, win_value=90.,
               enrich=192, next_width=4, free_plays=True)
    save_model(args.out / "challenger.json.gz", {**parent, "public_belief": cfg})
    report["panel_start"] = 192000000
    report["test"] = panel_chunks(args.out, "two_plays", w, 192000000, 32768, args.threads)
    write(location, report)


if __name__ == "__main__":
    main()
