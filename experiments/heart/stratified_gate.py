"""Optimize a draft policy against observed Heart entries using stratified ordinary starts."""
import argparse
import hashlib
import json
from pathlib import Path

import heart_native as engine
import numpy as np

from .deck_training import funnel
from .heart_curriculum import read_model, save_model
from .replay import apply_model, heart_encounters
from .train import statistics, write


def allocation(population, budget):
    population = np.asarray(population, dtype=int)
    if budget > population.sum() or budget < np.count_nonzero(population):
        raise ValueError("invalid stratified budget")
    late = min(population[3], budget//3)
    rest = budget-late
    n = np.array([round(.29*rest), round(.47*rest), 0, late], dtype=int)
    n[2] = budget-n.sum()
    n = np.minimum(population, np.maximum(n, population > 0))
    while n.sum() < budget:
        n[np.argmax(population-n)] += 1
    while n.sum() > budget:
        i = np.argmax(n-(population > 0))
        n[i] -= 1
    return n


def sampling_design(floors, start, budget, seed):
    g = np.digitize(floors,(18,35,50))
    sizes = np.bincount(g, minlength=4)
    n = allocation(sizes, budget)
    random = np.random.default_rng(seed)
    selected = np.concatenate([random.choice(np.flatnonzero(g == i), k, replace=False)
                               for i,k in enumerate(n)])
    return start+selected, g[selected], sizes


def paired_metric(data, original, strata, sizes):
    gain = se2 = rate = boss = act3 = 0.
    by_stage = []
    for stage, size in enumerate(sizes):
        if not size:
            continue
        x, y = data[strata == stage], original[strata == stage]
        w = size/sum(sizes)
        d = x[:,6]-y[:,6]
        rate += w*x[:,6].mean()
        gain += w*d.mean()
        se2 += w*w*(1-len(x)/size)*(d.var(ddof=1)/len(x) if len(x)>1 else 0.)
        boss += w*(x[:,0]>=50).mean()
        act3 += w*(x[:,1]>=3).mean()
        by_stage.append(dict(population=int(size), sampled=len(x), hits=int(x[:,6].sum()),
                             gain=int((d>0).sum()), lost=int((d<0).sum())))
    return dict(weighted_rate=rate, paired_gain=gain, paired_se=float(np.sqrt(se2)), act3=act3,
                first_boss=boss, raw_entries=int(data[:,6].sum()), stages=by_stage)


def rank(candidate):
    s = candidate["metrics"]
    # Reduce the chance that one high-weight easy-to-miss seed decides a generation.
    return (s["paired_gain"]-.7*s["paired_se"], s["first_boss"], s["act3"])


def design_check():
    result = {}
    for name,start,arm in (("paired-inventory-search",194000000,"combined"),("human-strategy",214000000,"context"),
                           ("stacked-search",216000000,"human")):
        d = Path("build")/name
        report = json.loads((d/"report.json").read_text())
        full = report[str(start)][arm] if name=="stacked-search" else report["test"][arm]
        floors = np.load(d/f"{start}-parent.npy")[:,0]
        g = np.digitize(floors,(18,35,50))
        sizes = np.bincount(g, minlength=4)
        hits = np.bincount(g[np.asarray(full["heart_encounter_seeds"],dtype=int)-start],minlength=4)
        n = allocation(sizes,1536)
        probs = hits/sizes
        q = hits.sum()/len(g)
        v = np.sum((sizes/len(g))**2 * (1-n/sizes)*probs*(1-probs)*sizes/(sizes-1)/n)
        uniform = (1-1536/len(g))*q*(1-q)*len(g)/(len(g)-1)/1536
        result[name] = dict(population=sizes.tolist(), positives=hits.tolist(), sampled=n.tolist(),
                            expected_stratified_labels=float(n@probs), expected_uniform_labels=1536*q,
                            finite_population_variance_ratio=float(v/uniform))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", type=Path, default=Path("build/stratified-gate"))
    args = p.parse_args()
    out = args.out
    out.mkdir(parents=True,exist_ok=True)
    path = out/"report.json"
    r = json.loads(path.read_text()) if path.exists() else dict(design_check=design_check(), generations=[], test={})
    parent = read_model("build/trajectory-teacher/challenger.json.gz")
    human = read_model("build/human-strategy/challenger.json.gz")
    zero = np.zeros(12)
    def scout(start):
        f = out/f"scout-{start}.npy"
        if not f.exists():
            w = apply_model("first",parent)
            np.save(f,np.asarray(engine.batch(w,start,32768,args.threads))[:,0])
        return np.load(f)
    train = scout(220000000)
    valid = scout(221000000)
    w = apply_model("first",human)
    def batch(params,seeds):
        engine.set_historical_gate(params.tolist())
        return np.asarray(engine.batch_seeds(w,seeds.tolist(),args.threads))
    if "parity" not in r:
        ids = np.arange(214000000,214000064)
        a = batch(zero,ids)
        if not np.array_equal(a[:,:6],engine.batch(w.tolist() if isinstance(w,np.ndarray) else w,int(ids[0]),len(ids),args.threads)):
            raise ValueError("arbitrary-seed runs differ from contiguous evaluation")
        wins = batch(zero,np.array([214004505,214004761]))
        if not (wins[:,4]>0).all() or not (wins[:,6]>0).all():
            raise ValueError("known Heart wins failed the entry flag")
        r["parity"] = dict(matched_runs=len(ids), checked_wins=2)
    initial_sd = np.array([.95]*5+[1.15]*4+[5.]*3)
    bounds = np.array([2.5]*5+[3.]*4+[15.]*3)
    center, sd = zero.copy(), initial_sd.copy()
    for generation in range(5):
        if generation < len(r["generations"]):
            record = r["generations"][generation]
            center,sd = np.array(record["next_center"]),np.array(record["next_sd"])
            continue
        seeds,strata,sizes = sampling_design(train,220000000,1536,22033+generation)
        reference = batch(zero,seeds)
        rng = np.random.default_rng(22065+generation)
        noise = rng.normal(size=(6,12))
        candidates = np.vstack([center, zero, center+sd*noise, center-sd*noise]).clip(-bounds,bounds)
        tested = []
        for params in np.unique(candidates,axis=0):
            rows = reference if np.all(params == 0) else batch(params,seeds)
            tested.append(dict(params=params.tolist(), metrics=paired_metric(rows,reference,strata,sizes)))
        tested.sort(key=rank,reverse=True)
        elite = np.asarray([c["params"] for c in tested[:4]])
        center = .25*center+.75*elite.mean(0)
        sd = np.maximum(.25*initial_sd,.35*sd+.65*elite.std(0))
        r["generations"].append(dict(generation=generation+1, candidates=tested,next_center=center.tolist(), next_sd=sd.tolist()))
        write(path,r)
        print("generation",generation+1,tested[0]["metrics"]["raw_entries"],flush=True)
    if "selection" not in r:
        seeds, strata, sizes = sampling_design(valid,221000000,4096,22153)
        reference = batch(zero,seeds)
        pool = [c["params"] for gen in r["generations"] for c in gen["candidates"][:3]]
        alternatives = []
        for params in np.unique(np.vstack([pool,center]),axis=0):
            if not params.any():
                continue
            alternatives.append(dict(params=params.tolist(), metrics=paired_metric(batch(params,seeds),reference,strata,sizes)))
        alternatives.sort(key=rank,reverse=True)
        r["selection"] = alternatives
        r["selected"] = alternatives[0]["params"]
        write(path,r)
    challenger = {**human,"historical_gate":r["selected"]}
    save_model(out/"challenger.json.gz",challenger)
    for name,m in (("parent",parent),("human",human),("gate",challenger)):
        if name in r["test"]:
            continue
        w = apply_model("first",m)
        seeds = np.arange(222000000,222032768)
        panels = [np.asarray(engine.batch_seeds(w,part.tolist(),args.threads)) for part in np.array_split(seeds,32)]
        rows = np.concatenate(panels)
        hits = seeds[rows[:,6] > 0].tolist()
        if hits != heart_encounters(w,int(seeds[0]),rows[:,:6]):
            raise ValueError("live entry flags differ from replay")
        np.save(out/f"222000000-{name}.npy",rows)
        digest = hashlib.sha256(json.dumps(m,separators=(",",":"),sort_keys=True).encode()).hexdigest()
        r["test"][name] = {**statistics(rows[:,:6]), **funnel(rows), "heart_encounter_seeds":hits,
                           "policy_sha256":digest}
        write(path,r)
        print(name,len(hits),flush=True)


if __name__=="__main__":
    main()
