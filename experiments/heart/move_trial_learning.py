"""Estimate the gain of replacing the search move, using actual paired continuations."""
import gzip
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from .afterstate_learning import export
from .combat_afterstate import estimated_difference


def summarize(pick,delta,seeds):
    changes = pick.astype(float) * delta
    unique,groups=np.unique(seeds,return_inverse=True)
    totals=np.bincount(groups,weights=changes)
    counts=np.bincount(groups)
    rng=np.random.default_rng(280843)
    draws=rng.integers(0,len(unique),(1500,len(unique)))
    boots=totals[draws].sum(1) / counts[draws].sum(1)
    return dict(states=len(delta),games=len(unique),replaced=int(pick.sum()),
                mean_gain=float(changes.mean()),random_same_fraction=float(pick.mean()*delta.mean()),
                gain_interval=np.quantile(boots,[.025,.975]).tolist())


def main():
    out=Path("build/move-trials")
    rows={k:[] for k in ("train","validation","test")}
    with gzip.open("build/public-expert-traces/replays.jsonl.gz","rt",encoding="utf8") as fp:
        expected=[]
        for i,line in enumerate(fp):
            if i>=400:break
            r=json.loads(line)
            if not r["error"] and r["public_information"] and r["original_audit"]["passed"] and r["fights"]:
                expected.append(out/f"game-{i}.json")
    missing=[p for p in expected if not p.exists()]
    if missing:raise ValueError(f"waiting for {len(missing)} of {len(expected)} games")
    games=0
    for path in expected:
        g=json.loads(path.read_text());games+=1
        for r in g["rows"]:
            rows[g["split"]].append((r["x"],np.mean(r["delta"]),g["seed"]))
    def data(k):
        x,y,g=zip(*rows[k])
        return np.array(x),np.array(y),np.array(g)
    x,y,_=data("train")
    xval,yval,gval=data("validation")
    mean=x.mean(0);sd=np.maximum(x.std(0),.05)
    models={}
    for ridge in (16.,128.):
        reg=Ridge(alpha=ridge).fit((x-mean)/sd,y)
        w=reg.coef_/sd
        policy=dict(kind="linear",bias=float(reg.intercept_-w@mean),coefficients=w.tolist())
        models["ridge"+str(int(ridge))]=policy
    gb=HistGradientBoostingRegressor(max_iter=48,max_leaf_nodes=7,min_samples_leaf=32,
                                     learning_rate=.06,l2_regularization=16,early_stopping=False,random_state=280771)
    gb.fit(x,y)
    models["trees"]=dict(kind="tree",**export(gb))
    vals={}
    candidates=[]
    for name,policy in models.items():
        pred=np.array([estimated_difference(policy,obs) for obs in xval])
        vals[name]={}
        for threshold in (0.,1.,3.,6.):
            s=summarize(pred>threshold,yval,gval);vals[name][str(threshold)]=s
            candidates.append((s["mean_gain"],-s["replaced"],name,threshold))
    _,_,selected,threshold=max(candidates)
    policy={**models[selected],"threshold":threshold}
    xt,yt,gt=data("test")
    pred=np.array([estimated_difference(policy,obs) for obs in xt])
    result=dict(source_games=games,train_states=len(y),validation=vals,selected=selected,threshold=threshold,
                test=summarize(pred>threshold,yt,gt),always_swap=summarize(np.ones(len(yt),bool),yt,gt),
                original_critic=summarize(1.5*xt[:,14] > np.maximum(0.,-xt[:,10])+1.,yt,gt))
    (out/"policy.json").write_text(json.dumps(policy),encoding="utf8")
    (out/"fit.json").write_text(json.dumps(result,indent=2),encoding="utf8")
    print(selected,threshold,result["test"],flush=True)


if __name__=="__main__":
    main()
