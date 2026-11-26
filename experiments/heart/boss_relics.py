"""Learn first-boss relic preferences and compare them on identical fresh A20 starts."""
import argparse
import gzip
import json
from pathlib import Path

import heart_native as engine
import numpy as np
import sts_env as sts

from .replay import REFERENCE, load_model
from .train import statistics, write


def compare(baseline, challenger):
    old = baseline[:, 1] >= 3
    new = challenger[:, 1] >= 3
    return {
        "gained_act3": int((new & ~old).sum()),
        "lost_act3": int((old & ~new).sum()),
        "changed_act2": int(np.count_nonzero((baseline[:, 1] >= 2) != (challenger[:, 1] >= 2))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=int, default=12288)
    parser.add_argument("--validation", type=int, default=4096)
    parser.add_argument("--test", type=int, default=32768)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--out", type=Path, default=Path("build/first-boss-relics"))
    args = parser.parse_args()
    if min(args.train, args.validation, args.test, args.threads) <= 0:
        parser.error("counts must be positive")
    if max(args.train, args.validation) > 1000000:
        parser.error("seed ranges must be disjoint")
    args.out.mkdir(exist_ok=True, parents=True)
    weights = load_model("first")
    train_start, validation_start, test_start = 53000000, 54000000, 55001000
    examples = np.asarray(engine.practice_first_boss(weights, train_start, args.train, args.threads))
    if not len(examples):
        raise RuntimeError("No eligible Act 1 chest in the training sample")
    offsets, ids, differences = examples.T
    counts = np.bincount(ids, minlength=engine.relic_count)
    totals = np.bincount(ids, weights=differences, minlength=engine.relic_count)
    # Thirty zero-advantage pseudo-observations temper noisy estimates for rare offers.
    values = totals / (counts + 30)
    report = dict(train_start=train_start, validation_start=validation_start, test_start=test_start,
                  train_runs=args.train, chests=int(len(np.unique(offsets))), offers=len(examples),
                  estimates={sts.RELIC_NAMES[i]: dict(count=int(counts[i]), advantage=float(values[i]))
                             for i in np.flatnonzero(counts)}, validation={})
    print(f"Fitted {len(examples)} alternatives from {report['chests']} chests.", flush=True)

    def panel(scale, start, n):
        engine.set_first_boss_bonus((scale * values).tolist())
        return np.asarray(engine.batch(weights, start, n, args.threads))

    # Select a challenger even when the frozen actor wins validation, so a negative result is measurable.
    best_rank = None
    for scale in (0, 1, 4, 16):
        rows = panel(scale, validation_start, args.validation)
        stats = statistics(rows)
        report["validation"][scale] = stats
        rank = (stats["act3"], int(rows[:, 0].sum()), -scale)
        if scale and (best_rank is None or rank > best_rank):
            chosen, best_rank = scale, rank
        print(f"Validation scale {scale}: {stats}", flush=True)
    report["selected_scale"] = chosen
    # Freeze the artifact before touching the final panel. The ordinary evaluate CLI can load it directly.
    with gzip.open(REFERENCE/"checkpoint.json.gz", "rt", encoding="utf8") as source:
        model = json.load(source)
    model["first_boss_relics"] = (chosen * values).tolist()
    with gzip.open(args.out/"challenger.json.gz", "wt", encoding="utf8") as dest:
        json.dump(model, dest)
    baseline = panel(0, test_start, args.test)
    challenger = panel(chosen, test_start, args.test)
    report["test"] = dict(first=statistics(baseline), challenger=statistics(challenger),
                          paired=compare(baseline, challenger))
    write(args.out/"report.json", report)
    print(report["test"], flush=True)


if __name__ == "__main__":
    main()
