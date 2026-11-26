"""Measure the value of arriving at an earned boss with more HP, holding the deck and seed fixed."""
import argparse
from pathlib import Path

import heart_native as engine
import numpy as np

from .heart_curriculum import read_model
from .matched_fights import SETTINGS, configure
from .replay import apply_model
from .train import write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("build/boss-entry-health"))
    p.add_argument("--threads", type=int, default=3)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    parent = read_model("build/trajectory-teacher/challenger.json.gz")
    w = apply_model("first", parent)
    # This predeclared scout contains only ordinary, unmodified baseline starts.
    scout = np.load("build/search-components/scout.npy")
    engine.collect_beliefs(w,190000000,3072)
    rng = np.random.default_rng(22003)
    report = {"scout_start":207000000, "scouts":len(scout), "floors":{}}
    for floor in (32,49):
        qualified = np.flatnonzero(scout[:,0] >= floor)
        ids = np.sort(rng.choice(qualified, min(160,len(qualified)), replace=False))
        result = report["floors"][floor] = {"eligible":len(qualified), "sampled_seeds":(207000000+ids).tolist(), "arms":{}}
        for name in ("parent","current_search"):
            arms = result["arms"][name] = {}
            raw = {}
            for tag,bonus in (("natural",0),("plus12",12),("full",-1)):
                path = args.out / f"{floor}-{name}-{tag}.npy"
                if not path.exists():
                    batches = []
                    for offset in range(0,len(ids),20):
                        configure(SETTINGS["parent"])
                        seeds = 207000000+ids[offset:offset+20]
                        bank = engine.Bank(w,seeds.tolist(),floor,args.threads)
                        configure(SETTINGS[name])
                        batches.append(np.asarray(bank.entry_health(w,bonus,args.threads)))
                    np.save(path,np.concatenate(batches))
                raw[tag] = np.load(path)
            ok = raw["natural"][:,0] >= 0
            if not np.any(ok):
                raise RuntimeError("failed to identify pre-boss map choices")
            for tag,v in raw.items():
                if not np.array_equal(v[:,:3],raw["natural"][:,:3]):
                    raise RuntimeError("unmatched boss, initial HP, or maximum HP")
                s = v[ok]
                base = raw["natural"][ok,5] > 0
                new = s[:,5] > 0
                arms[tag] = dict(fights=len(s), missing=int((~ok).sum()), wins=int(new.sum()),
                                 new_wins=int((new & ~base).sum()), lost_wins=int((~new & base).sum()),
                                 mean_hp_before=float(s[:,1].mean()), mean_assigned_hp=float(s[:,3].mean()),
                                 mean_exit_hp=float(s[:,6].mean()), capped=int(s[:,9].sum()))
            if name == "parent":
                actual = scout[ids[ok],0] > floor+1
                if not np.array_equal(actual,raw["natural"][ok,5] > 0):
                    raise RuntimeError("unchanged fight does not reproduce ordinary progress past the boss")
            write(args.out / "report.json",report)
            print(floor,name,arms,flush=True)


if __name__ == "__main__":
    main()
