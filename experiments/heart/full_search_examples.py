"""Recover full public observations for the exact decisions used by the compact search student."""
import argparse
import json
from pathlib import Path

import heart_native as engine
import numpy as np
import sts_env as sts

from .heart_curriculum import read_model
from .replay import apply_model


def capture(path, destination, weights, threads):
    with np.load(path) as archived:
        rows = archived["rows"]
    groups = np.split(rows, 1 + np.flatnonzero(np.any(rows[1:, :2] != rows[:-1, :2], axis=1)))
    n = 32
    start = int(path.stem)
    lookup = {(int(g[0, 0]), int(g[0, 1])): (i, g) for i, g in enumerate(groups)}
    last = np.full(n, -1)
    for episode, step in lookup:
        last[episode] = max(last[episode], step)
    env = sts.VecEnv(n, 20, threads, 3500, True, False, "ironclad", 0., 0, 0.)
    seeds = np.arange(start, start + n)
    env.reset(seeds.tolist())
    observations = np.empty((len(groups), sts.OBS_SIZE), np.float32)
    prior = np.full((len(groups), sts.N_ACTIONS), -1e8, np.float32)
    teacher = prior.copy()
    count = 0
    for step in range(int(last.max()) + 1):
        obs, mask = np.asarray(env.obs), np.asarray(env.mask)
        active = last >= step
        if not np.array_equal(np.asarray(env.seed)[active], seeds[active]):
            raise RuntimeError("a practice run ended before its recorded decision")
        actions = mask.argmax(axis=1).astype(np.int32)
        free = active & (mask.sum(axis=1) > 1)
        if free.any():
            actions[free] = engine.teacher_batch(weights, obs[free], mask[free], threads)
        for episode in np.flatnonzero(active):
            sample = lookup.get((episode, step))
            if sample is None:
                continue
            i, group = sample
            legal = group[:, 2].astype(int)
            winner = group[group[:, 6] != 0, 2]
            if len(winner) != 1 or actions[episode] != winner[0] or not mask[episode, legal].all():
                raise RuntimeError(f"teacher replay diverged on seed {start + episode}, step {step}")
            observations[i] = obs[episode]
            prior[i, legal] = group[:, 7]
            teacher[i, legal] = group[:, 7] + group[:, 5]
            count += 1
        if env.step(actions):
            raise RuntimeError("illegal replay move")
    if count != len(groups):
        raise RuntimeError("incomplete observation capture")
    tmp = destination.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, obs=observations, prior=prior, teacher=teacher,
                        seeds=np.array([start + int(g[0, 0]) for g in groups]))
    tmp.replace(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("build/full-search-student"))
    parser.add_argument("--threads", type=int, default=3)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source = Path("build/belief-student")
    parent = read_model("build/trajectory-teacher/challenger.json.gz")
    belief = json.loads(Path("build/public-belief/report.json").read_text())["selected"]
    weights = apply_model("first", {**parent, "public_belief": belief})
    for path in sorted(source.glob("*.npz")):
        if not path.stem.isdigit():
            continue
        dest = args.out / path.name
        if not dest.exists():
            capture(path, dest, weights, args.threads)
            print("captured", dest.name, flush=True)


if __name__ == "__main__":
    main()
