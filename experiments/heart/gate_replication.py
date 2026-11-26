"""Replicate the frozen Heart-trained policy and isolate its three adjustments."""
import argparse
from pathlib import Path

import heart_native as engine
import numpy as np

from .deck_training import funnel
from .heart_curriculum import read_model
from .replay import apply_model, heart_encounters
from .train import statistics, write


def evaluate(out, start, name, model, threads):
    path = out / f"{start}-{name}.npy"
    weights = apply_model("first", model)
    if path.exists():
        rows = np.load(path)
    else:
        chunks = np.array_split(np.arange(start, start + 32768), 32)
        rows = np.concatenate([engine.batch_seeds(weights, part.tolist(), threads) for part in chunks])
        np.save(path, rows)
    hits = (np.flatnonzero(rows[:, 6]) + start).tolist()
    if hits != heart_encounters(weights, start, rows[:, :6]):
        raise RuntimeError("recorded entries differ from the replay")
    return {**statistics(rows[:, :6]), **funnel(rows), "heart_encounter_seeds": hits}


def main():
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=3)
    args = parser.parse_args()
    out = Path("build/stratified-gate")
    fitted = read_model(out / "challenger.json.gz")
    human = read_model("build/human-strategy/challenger.json.gz")
    params = fitted["historical_gate"]
    arms = {"human": human, "gate": fitted}
    for name, part in (("act", slice(0, 5)), ("resources", slice(5, 9)), ("take_card", slice(9, 12))):
        coefficients = [0.] * 12
        coefficients[part] = params[part]
        arms[name] = {**human, "historical_gate": coefficients}
    path = out / "replication.json"
    report = json.loads(path.read_text()) if path.exists() else {}
    for start in (223000000, 222000000):
        panel = report.setdefault(str(start), {})
        for name, model in arms.items():
            if name in panel:
                continue
            panel[name] = evaluate(out, start, name, model, args.threads)
            write(path, report)
            print(start, name, len(panel[name]["heart_encounter_seeds"]), flush=True)


if __name__ == "__main__":
    main()
