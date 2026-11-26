"""Choose an entire downstream strategy once, from the public Act 2 opening."""
import argparse
from hashlib import sha256
from pathlib import Path

import heart_native as engine
import numpy as np

from .heart_curriculum import read_model, save_model
from .replay import apply_model
from .train import statistics, write


PATHS = {
    "reference": "build/first-boss-relics/challenger.json.gz",
    "staged": "build/heart-curriculum/staged.json.gz",
    "joint": "build/joint-strategy/challenger.json.gz",
    "deeper_search": "build/turn-depth/challenger.json.gz",
}


def collect(models, start, count, threads, gate=None, audit=False):
    """During deployment each surviving run plays exactly one chosen suffix."""
    observed, paired, episodes, selected = [], [], [], []
    for offset in range(0, count, 512):
        batch = engine.ActOpening(apply_model("first", models[0]), start + offset,
                                  min(512, count-offset), threads)
        initial = np.asarray(batch.prefix)
        ids = np.asarray(batch.arrivals, dtype=int)
        outcomes = initial.copy()
        if len(ids):
            features = np.asarray(batch.features())
            if gate is not None:
                choices = predict(gate, features)
                for index in np.unique(choices):
                    group = ids[choices == index]
                    weights = apply_model("first", models[index])
                    outcomes[group] = batch.finish(weights, group.tolist(), threads)
                selected.extend(choices.tolist())
            if gate is None or audit:
                alternatives = []
                for model in models:
                    weights = apply_model("first", model)
                    alternatives.append(batch.finish(weights, ids.tolist(), threads))
                alternatives = np.stack(alternatives, axis=1)
                if audit:
                    np.testing.assert_array_equal(outcomes[ids], alternatives[np.arange(len(ids)), choices])
                observed.append(features)
                paired.append(alternatives)
        episodes.append(outcomes)
    return (np.concatenate(observed) if observed else None,
            np.concatenate(paired) if paired else None,
            np.concatenate(episodes), np.asarray(selected, dtype=int))


def utility(rows):
    floor, act, encounter = (rows[..., i] for i in range(3))
    return floor/16 + (act >= 3) + 2*(floor >= 49) + 3*(act >= 4) + 15*encounter


def predict(gate, features):
    if "constant" in gate:
        return np.full(len(features), gate["constant"], dtype=int)
    return (features @ np.asarray(gate["weights"]) + gate["bias"]).argmax(axis=1)


def candidates(features, rows):
    scores = utility(rows)
    # Subtracting the shared position value leaves the differences the gate
    # can influence. Shrink coefficients on rare card and relic identities.
    scores -= scores.mean(axis=1, keepdims=True)
    bias = scores.mean(axis=0)
    mean = features.mean(axis=0)
    scale = features.std(axis=0).clip(.2)
    x = (features-mean)/scale
    active = (x*x).sum(axis=0) > 1e-6
    design = x[:,active]
    gram = design.T @ design
    rhs = design.T @ (scores-bias)
    for penalty in (30, 300, 1500):
        weights = np.zeros((features.shape[1], scores.shape[1]))
        weights[active] = np.linalg.solve(gram+np.eye(len(gram))*penalty, rhs)/scale[active,None]
        yield dict(penalty=penalty, weights=weights.tolist(), bias=(bias-mean@weights).tolist())


def summarize(rows):
    return dict(**statistics(rows), encounters=int(rows[:,2].sum()))


def report_gate(gate, features, rows):
    choices = predict(gate, features)
    chosen = rows[np.arange(len(rows)), choices]
    return dict(mean_utility=float(utility(chosen).mean()), choices=np.bincount(
        choices, minlength=rows.shape[1]).tolist(), **summarize(chosen))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("build/stage-policy"))
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--replay-start", type=int, help="Evaluate an already frozen gate from fresh starts")
    parser.add_argument("--runs", type=int, default=32768)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    artifact = args.out/"selector.json.gz"
    models = [read_model(path) for path in PATHS.values()]
    hashes = {name:sha256(Path(path).read_bytes()).hexdigest() for name,path in PATHS.items()}
    if args.replay_start is not None:
        frozen = read_model(artifact)
        if hashes != frozen["sha256"] or list(PATHS) != frozen["order"]:
            raise ValueError("selector and downstream policies differ")
        _,_,rows,_ = collect(models,args.replay_start,args.runs,args.threads,frozen["gate"])
        print(summarize(rows))
        return

    x_train,y_train,_,_ = collect(models,94000000,12288,args.threads)
    print("paired Act 2 training positions",len(x_train),flush=True)
    x_val,y_val,_,_ = collect(models,95000000,4096,args.threads)
    gates = list(candidates(x_train,y_train))
    val = [report_gate(gate,x_val,y_val) for gate in gates]
    constant = [report_gate({"constant":i},x_val,y_val) for i in range(len(models))]
    winner = int(np.argmax([s["mean_utility"] for s in val]))
    constant_id = int(np.argmax([s["mean_utility"] for s in constant]))
    gate = gates[winner]
    save_model(artifact,dict(gate=gate,sha256=hashes,order=list(PATHS)))
    print("frozen gate",gate["penalty"],"constant",list(PATHS)[constant_id],flush=True)
    start = 96000000
    _,alternatives,rows,choices = collect(models,start,args.runs,args.threads,gate,audit=True)
    # On test the audit runs after routing. It measures the ceiling of this
    # particular menu, even for a selector that knew every future in advance.
    arrivals = np.flatnonzero(rows[:,1]>=2)
    fixed = {}
    for i,name in enumerate(PATHS):
        split = rows.copy()
        split[arrivals] = alternatives[:,i]
        fixed[name] = summarize(split)
    report = dict(train_start=94000000,training_runs=12288,training_positions=len(x_train),
                  validation_start=95000000,validation_runs=4096,validation_positions=len(x_val),
                  policies=list(PATHS),validation=val,constant_validation=constant,selected_penalty=gate["penalty"],
                  selected_constant=list(PATHS)[constant_id],test_start=start,hashes=hashes,
                  selector=summarize(rows),choices=np.bincount(choices,minlength=len(models)).tolist(),
                  encounter_seeds=(start+np.flatnonzero(rows[:,2])).tolist(),fixed_suffix=fixed,
                  any_suffix_encounter=int(alternatives[:,:,2].any(axis=1).sum()),
                  any_suffix_act3=int((alternatives[:,:,1]>=3).any(axis=1).sum()))
    write(args.out/"report.json",report)
    print(report["selector"],"any suffix encountered",report["any_suffix_encounter"],flush=True)


if __name__ == "__main__":
    main()
