"""Train a combat value function on verified fights; select tree size on unseen games."""
import gzip
import json
from pathlib import Path
import re

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from .combat_afterstate import estimate, features


def dataset(root, layout, splits, out):
    size = layout["size"]
    dt = np.dtype([("seed", "<u8"), ("step", "<u4"), ("obs", "<i2", size),
                   ("legal", "u1", 81), ("action", "u1")])
    with gzip.open("build/public-expert-traces/replays.jsonl.gz", "rt", encoding="utf8") as fp:
        outcomes = {}
        for game in map(json.loads, fp):
            for f in game["fights"]:
                key = (int(game["seed"]), f["act"], f["floor"])
                # Two distinct fights can take place on the same event floor.
                outcomes[key] = None if key in outcomes else (f["survived"], f["exit_hp"])
    train = np.memmap(root / "obs-train.bin", mode="r", dtype=dt)
    for zone, key, limit in (("hand","critic_cards",48),("monsters","critic_enemies",64)):
        start, count, width = layout["zones"][zone]
        ids = train["obs"][:, start:start+count*width:width]
        freqs = np.bincount(ids.flatten())
        layout[key] = sorted(int(i) for i in np.argsort(-freqs[1:],kind="stable")[:limit] + 1 if freqs[i])
    text=Path("build/rl-fork/include/constants/Relics.h").read_text(encoding="utf8")
    relic_names=re.findall(r'"([^"]*)"',text.split("relicIds[]",1)[1].split("};",1)[0])
    layout["frozen_eye"] = relic_names.index("Frozen Eye")
    for split in splits:
        file = out / (split + ".npz")
        if file.exists():
            continue
        mem = np.memmap(root / f"obs-{split}.bin", mode="r", dtype=dt)
        xs, target, groups = [], [], []
        indexes = {}
        for example in mem:
            o = example["obs"]
            key = (int(example["seed"]), int(o[4]), int(o[3]))
            label = outcomes.get(key)
            if label is None:
                continue
            xs.append(features(o.tolist(), layout))
            win, health = label
            target.append(35. * win + max(0, health) * win)
            groups.append(indexes.setdefault(key, len(indexes)))
        np.savez_compressed(file, x=np.asarray(xs,dtype=np.float32), y=target, group=groups)
        print(split, len(xs), len(indexes), flush=True)


def weights(groups):
    # A long fight generates more rows. Give each fight equal total weight.
    w = 1. / np.bincount(groups)[groups]
    return w / w.mean()


def metrics(pred, data):
    y = data["y"]
    w = weights(data["group"])
    return dict(states=len(y), fights=int(1 + max(data["group"])),
                mae=float(np.average(np.abs(y-pred), weights=w)),
                rmse=float(np.sqrt(np.average((y-pred)**2, weights=w))))


def export(net):
    trees = []
    for (iteration,) in net._predictors:
        rows = iteration.nodes
        trees.append([[int(n["feature_idx"]),float(n["num_threshold"]),int(n["left"]),
                       int(n["right"]),float(n["value"])] for n in rows])
    return dict(bias=float(net._baseline_prediction.item()), trees=trees)


def main():
    out = Path("build/afterstate-critic")
    out.mkdir(parents=True,exist_ok=True)
    root = Path("build/public-expert-intent")
    layout = json.loads((root / "layout.json").read_text())
    splits = ("train", "validation", "test")
    dataset(root, layout, splits, out)
    train, val, test = [np.load(out / f"{s}.npz") for s in splits]
    fits = {}
    for name, width, leaves in (("health",8,15),("tactical",None,15),("wide",None,31)):
        net = HistGradientBoostingRegressor(max_iter=100,learning_rate=.11,l2_regularization=8.,
                                            max_leaf_nodes=leaves,early_stopping=False,random_state=277771)
        net.fit(train["x"][:,:width],train["y"],sample_weight=weights(train["group"]))
        fit = {"validation":metrics(net.predict(val["x"][:,:width]),val)}
        fits[name] = (net,width,fit)
    chosen = min(("tactical","wide"), key=lambda k: fits[k][2]["validation"]["rmse"])
    for name in ("health",chosen):
        net,width,fit=fits[name]
        fit["test"]=metrics(net.predict(test["x"][:,:width]),test)
    net = fits[chosen][0]
    policy = export(net)
    sample = test["x"][::97]
    error = float(max(abs(net.predict(sample)-[estimate(policy, x) for x in sample])))
    if error > 1e-7:
        raise ValueError(("portable tree export mismatch",error))
    policy.update(cards=layout["critic_cards"],enemies=layout["critic_enemies"],frozen_eye=layout["frozen_eye"])
    (out/"critic.json").write_text(json.dumps(policy),encoding="utf8")
    report = dict(models={k:v[2] for k,v in fits.items()},selected=chosen,export_max_error=error,
                  training_states=len(train["y"]),training_fights=int(max(train["group"])+1))
    (out/"fit.json").write_text(json.dumps(report,indent=2),encoding="utf8")
    print(report,flush=True)


if __name__ == "__main__":
    main()
