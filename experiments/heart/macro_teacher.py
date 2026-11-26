"""Collect offline labels from repeated legal macro choices with a fixed combat policy."""
import argparse
from pathlib import Path

import heart_native as engine
import numpy as np

from .heart_curriculum import read_model
from .replay import apply_model
from .train import write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--runs", type=int, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("build/macro-teacher"))
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("runs must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    weights = apply_model("first", read_model("build/trajectory-teacher/challenger.json.gz"))
    batches = []
    for offset in range(0, args.runs, 32):
        seed = args.start+offset
        examples, stats = engine.macro_teacher(weights, seed, min(32,args.runs-offset), args.threads)
        stats = np.asarray(stats)
        if stats[:,9].any() or (stats[:,2]<stats[:,0]).any():
            raise RuntimeError(f"continuation replay diverged on batch {seed}")
        np.savez_compressed(args.out/f"{seed}.npz", examples=examples, stats=stats)
        batches.append(stats)
        combined = np.concatenate(batches)
        summary = dict(start=args.start, runs=len(combined),
                       ordinary_heart=int(combined[:,1].sum()),teacher_heart=int(combined[:,3].sum()),
                       teacher_wins=int(combined[:,4].sum()),truncated=int(combined[:,5].sum()),
                       ordinary_mean_floor=float(combined[:,0].mean()),teacher_mean_floor=float(combined[:,2].mean()),
                       branches=int(combined[:,7].sum()),decisions=int(combined[:,6].sum()),
                       overrides=int(combined[:,8].sum()),replay_mismatches=int(combined[:,9].sum()))
        write(args.out/f"summary-{args.start}.json",summary)
        print(summary,flush=True)


if __name__ == "__main__":
    main()
