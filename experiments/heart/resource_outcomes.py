"""Compare HP and potion interventions against the unchanged continuation on the same earned menu."""
from pathlib import Path

import heart_native as engine
import numpy as np

from .heart_curriculum import read_model
from .replay import apply_model
from .train import write


def main():
    out = Path("build/resource-effects")
    out.mkdir(parents=True, exist_ok=True)
    parent = read_model("build/route-irl/selected-whole_map.json.gz")
    w = apply_model("first", parent)
    reference = np.load("build/map-preferences/scout.npy")
    rng = np.random.default_rng(268511)
    report = {}
    for act in (3, 2, 1):
        possible = np.flatnonzero((reference[:, 1] >= act) & (reference[:, 5] == 0))
        sample = np.sort(rng.choice(possible, min(3500, len(possible)), replace=False))
        used = groups = rows = 0
        for start in range(0, len(sample), 128):
            path = out / f"act-{act}-{start}.npy"
            ix = sample[start:start + 128]
            if not path.exists():
                data = engine.resource_outcomes(w, (251000000 + ix).tolist(), act, 3)
                result = []
                for i, values in enumerate(data):
                    if not values:
                        continue
                    a = np.asarray(values, dtype=np.int32)
                    control = a[a[:, 9] == 1]
                    if len(control) != 1 or not np.array_equal(control[0, 13:19], reference[ix[i], [0, 1, 2, 6, 4, 5]]):
                        raise ValueError(f"unchanged resource state diverged at seed {251000000 + ix[i]}")
                    if a[:, 18].any():
                        continue
                    a[:, 0] = ix[i]
                    result.append(a)
                np.save(path, np.concatenate(result) if result else np.empty((0, 19), dtype=np.int32))
            a = np.load(path)
            rows += len(a)
            groups += len(np.unique(a[:, 0]))
            used += int(a[:, 8].sum())
        report[act] = dict(attempts=len(sample), groups=groups, continuations=rows, potion_removals=used)
        write(out / "collection.json", report)
        print(act, report[act], flush=True)


if __name__ == "__main__":
    main()
