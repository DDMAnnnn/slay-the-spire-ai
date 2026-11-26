"""Compare visible-hand search depths on matched full runs with frozen weights."""
import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import heart_native as engine
import numpy as np

from .boss_relics import compare
from .replay import load_model
from .train import statistics


def panel(weights, depths, start, count, threads):
    rows = {depth: [] for depth in depths}
    seconds = dict.fromkeys(depths, 0.)
    # Rotate depth order across seed blocks to spread machine load and warm-up effects.
    for block, offset in enumerate(range(0, count, 512)):
        first = block % len(depths)
        for depth in depths[first:] + depths[:first]:
            engine.set_turn_depth(depth)
            begin = perf_counter()
            batch = engine.batch(weights, start + offset, min(512, count - offset), threads)
            seconds[depth] += perf_counter() - begin
            rows[depth].extend(batch)
    arrays = {depth: np.asarray(batch) for depth, batch in rows.items()}
    results = {}
    for depth, array in arrays.items():
        results[depth] = dict(**statistics(array), seconds=round(seconds[depth], 3),
                              **compare(arrays[2], array))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("build/first-boss-relics/challenger.json.gz"))
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("build/turn-depth"))
    parser.add_argument("--benchmark", action="store_true",
                        help="measure full-run throughput at every depth on a separate matched panel")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    weights = load_model("first", args.model)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.benchmark:
        results = panel(weights, [1, 2, 3, 4, 5], 58000000, 16384, args.threads)
        for result in results.values():
            result["act3_per_minute"] = round(60 * result["act3"] / result["seconds"], 1)
        report = dict(start=58000000, threads=args.threads, results=results,
                      checkpoint_sha256=sha256(args.model.read_bytes()).hexdigest())
        (args.out/"throughput.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf8", newline="\n")
        print("Throughput", results, flush=True)
        return
    validation = panel(weights, [1, 2, 3, 4, 5], 56000000, 8192, args.threads)
    print("Validation", validation, flush=True)
    # Test the strongest deeper challenger even if it loses to the original on validation.
    selected = max((3, 4, 5), key=lambda d: (validation[d]["act3"], -d))
    with gzip.open(args.model, "rt", encoding="utf8") as source:
        model = json.load(source)
    model["turn_depth"] = selected
    with gzip.open(args.out/"challenger.json.gz", "wt", encoding="utf8") as dest:
        json.dump(model, dest)
    test = panel(weights, [2, selected], 57000000, 32768, args.threads)
    report = dict(checkpoint_sha256=sha256(args.model.read_bytes()).hexdigest(),
                  variant="first", threads=args.threads, validation_start=56000000,
                  validation=validation, selected_depth=selected, test_start=57000000, test=test)
    (args.out/"report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf8", newline="\n")
    print("Held out", test, flush=True)


if __name__ == "__main__":
    main()
