"""Compare a public combat tree with a one-decision bandit on the same determinization budget."""
import argparse
import json
from pathlib import Path

import heart_native as engine
import numpy as np
import sts_env as sts

from .heart_curriculum import read_model, save_model
from .inventory_search import panel_chunks
from .matched_fights import SETTINGS, configure, summary
from .public_belief import evaluate
from .replay import apply_model, rows
from .train import write


VARIANTS = {"search": (0,1), "root64": (64,1), "tree64": (64,7), "tree128": (128,7)}


def enable(name):
    original = name == "search"
    configure(SETTINGS["current_search"] if original else (8,1,5.,1,90.,True))
    engine.set_continuation(0, not original)
    engine.set_belief_action_limit(0 if original else 7)
    engine.set_public_tree(*VARIANTS[name])


def audit(w):
    enable("tree64")
    seen = 0
    for seed in range(217000020,217000024):
        env = sts.Env()
        env.reset(seed,20,final_act=True,character="ironclad")
        while not env.done:
            o, m = env.observe(), env.mask()
            scores = np.asarray(engine.scores(w,o,m))
            if o[7] and m[sts.ACT_END_TURN]:
                other = o.copy()
                d = rows(other,"draw")
                k = int(np.count_nonzero(d[:,0]))
                if k > 1:
                    d[:k] = d[:k][::-1].copy()
                    if not np.array_equal(scores,engine.scores(w,other,m)):
                        raise ValueError("hidden draw order changed a tree score")
                    seen += 1
            if not env.step(int(scores.argmax())):
                raise ValueError("illegal action in tree audit")
    counts = engine.public_tree_audit(w,217000000,4)
    if not seen or not counts[3]:
        raise RuntimeError("no draw-order checks or revisited public branches")
    return dict(reversed_draw_checks=seen, searched=int(counts[0]), rollouts=int(counts[1]),
                successor_nodes=int(counts[2]), revisits=int(counts[3]), followup_revisits=int(counts[4]))


def fights(w,start,n,names,threads):
    out = {k: [] for k in names}
    for offset in range(0,n,16):
        engine.set_public_tree(0,1)
        configure(SETTINGS["parent"])
        bank = engine.MatchedFights(w,start+offset,min(16,n-offset),threads)
        for name in names:
            enable(name)
            out[name].append(np.column_stack((bank.info,bank.evaluate(w,threads))))
    return {k: summary(np.concatenate(v)) for k,v in out.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", type=Path, default=Path("build/public-tree"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    f = args.out / "report.json"
    r = json.loads(f.read_text()) if f.exists() else {}
    parent = read_model("build/trajectory-teacher/challenger.json.gz")
    w = apply_model("first",parent)
    engine.collect_beliefs(w,190000000,3072)
    engine.enrich_late_particles(w,192)
    if "audit" not in r:
        r["audit"] = audit(w)
        write(f,r)
    if "calibration" not in r:
        r["calibration"] = fights(w,217100000,96,VARIANTS,args.threads)
        def score(k):
            a = r["calibration"][k]
            return a["fatal"]["wins"]+2*a["control"]["wins"]+.1*a["control"]["mean_exit_hp"]
        r["chosen"] = max(("tree64","tree128"), key=score)
        write(f,r)
    chosen = r["chosen"]
    # Hold the root visit budget fixed when isolating the contribution of later decisions.
    name = "root" + str(VARIANTS[chosen][0])
    VARIANTS[name] = (VARIANTS[chosen][0],1)
    if "heldout" not in r:
        r["heldout"] = fights(w,217200000,128,("search",name,chosen),args.threads)
        write(f,r)
    screen = r.setdefault("screen", {})
    for variant in ("search",name,chosen):
        if variant not in screen:
            enable(variant)
            screen[variant] = evaluate(w,218000000,1024,args.threads)
            write(f,r)
    def rank(v):
        return v["first_boss"]+2*v["second_boss"]+.1*v["act3"]+4*v["act4"]
    if rank(screen[chosen]) <= max(rank(screen[k]) for k in ("search",name)):
        r["qualification"] = "Did not exceed both fixed comparators on the ordinary run screen"
        write(f,r)
        return
    budget, depth = VARIANTS[chosen]
    cfg = dict(seed=190000000,count=3072,draws=8,scope=1,scale=5.,plans=1,win_value=90.,enrich=192,
               action_limit=7,free_plays=True,tree_budget=budget,tree_depth=depth)
    save_model(args.out/"challenger.json.gz",{**parent,"public_belief":cfg})
    enable(chosen)
    r["panel_start"] = 192000000
    r["test"] = panel_chunks(args.out,chosen,w,192000000,32768,args.threads)
    write(f,r)


if __name__ == "__main__":
    main()
