"""Measure a root-local exact public-score cache with replay and full-run parity checks."""
import argparse
from pathlib import Path
from time import perf_counter

import heart_native as engine
import numpy as np

from .replay import load_model
from .train import write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--threads", type=int, default=3)
    p.add_argument("--out", type=Path, default=Path("build/rollout-cache.json"))
    args = p.parse_args()
    w = load_model("first", "build/matched-fights/challenger.json.gz")
    baseline = None
    times = {False: [], True: []}
    for enabled in (False, True, True, False):
        engine.set_rollout_cache(enabled)
        start = perf_counter()
        data = np.asarray(engine.batch(w,219000000,24,args.threads))
        times[enabled].append(perf_counter()-start)
        if baseline is None:
            baseline = data
        if not np.array_equal(data,baseline):
            raise AssertionError("cache changed the matched batch")
    comparisons = 0
    for seed in range(219000024,219000028):
        engine.set_rollout_cache(False)
        expected = engine.trace(w,seed)
        engine.set_rollout_cache(True)
        if engine.trace(w,seed) != expected:
            raise AssertionError("cache changed a decision")
        comparisons += len(expected)
    hits, misses = engine.rollout_cache_audit(w,219000028,4)
    result = dict(runs=24, repeats=2, seconds_without=times[False], seconds_with=times[True],
                  speedup=float(np.mean(times[False])/np.mean(times[True])), exact_trace_decisions=comparisons,
                  hits=hits, misses=misses)
    write(args.out,result)
    print(result)


if __name__ == "__main__":
    main()
