"""Evaluate all four Neow options during training and choose using only the visible offer."""
import argparse
import gzip
import json
from pathlib import Path

import heart_native as engine
import numpy as np

from .boss_relics import compare
from .replay import heart_encounters, load_model, set_opening
from .train import statistics, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, default=Path("build/first-boss-relics/challenger.json.gz"))
    parser.add_argument("--out", type=Path, default=Path("build/learned-opening"))
    parser.add_argument("--threads", type=int, default=10)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    w = load_model("first", args.prior)
    train = np.asarray(engine.neow_practice(w, 64000000, 16384, args.threads)).reshape(-1, 4, 7)
    if not np.all(train[:, :, :2] > 0):
        raise ValueError("Expected all four ordinary Neow offers")
    reward, cost = train[:, :, 0]-1, train[:, :, 1]-1
    counts = np.zeros(engine.neow_shape)
    np.add.at(counts, (reward, cost), 1)
    candidates = {}
    trials = {}
    for bonus in (0, 40, 160):
        returns = (train[:, :, 2] + 6*train[:, :, 3] + 2*train[:, :, 4]
                   + bonus*(train[:, :, 3]>=3) + 100*train[:, :, 5] - 8*train[:, :, 6])
        advantages = returns - returns[:, 3:4]
        sums = np.zeros_like(counts)
        np.add.at(sums, (reward, cost), advantages)
        candidates[bonus] = (sums / (counts + 32)).tolist()
        set_opening(candidates[bonus])
        trials[bonus] = statistics(np.asarray(engine.batch(w, 65000000, 4096, args.threads)))
    selected = max(trials, key=lambda b:(trials[b]["act3"], trials[b]["mean_floor"]))
    with gzip.open(args.prior, "rt", encoding="utf8") as f:
        model = json.load(f)
    model["neow"] = candidates[selected]
    with gzip.open(args.out/"challenger.json.gz", "wt", encoding="utf8") as f:
        json.dump(model, f)
    report = dict(train_start=64000000, train_runs=len(train), validation=trials, validation_start=65000000,
                  selected=selected, test_start=66000000, test={})
    for name, table in (("previous", None), ("opening", model["neow"])):
        set_opening(table)
        result=np.asarray(engine.batch(w, 66000000, 32768, args.threads))
        if name=="previous":
            baseline=result
        report["test"][name]=dict(**statistics(result), **compare(baseline,result),
                                 heart_encounter_seeds=heart_encounters(w,66000000,result))
        print(name,report["test"][name],flush=True)
    write(args.out/"report.json",report)


if __name__=="__main__":
    main()
