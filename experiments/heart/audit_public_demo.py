"""Compare cross-fork observation tensors on matching combats at the same archived decision."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import heart_native as engine
import numpy as np
import sts_env as sts

from .shared_combat import demo_rows, projected
from .train import write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("build/public-expert-traces"))
    p.add_argument("--source-root",type=Path)
    p.add_argument("--broad",action="store_true")
    args = p.parse_args()
    b = args.root
    source=args.source_root or b
    rich="rendering" in json.loads((b/"layout.json").read_text())
    splits = [demo_rows(b, s) for s in ("train","validation","test") if (b/f"obs-{s}.bin").exists()]
    indices = {}
    for data in splits:
        cuts = np.r_[0,1+np.flatnonzero(data["seed"][1:]!=data["seed"][:-1]),len(data)]
        for i,j in zip(cuts[:-1],cuts[1:]):
            indices[int(data["seed"][i])] = data[i:j]
    with gzip.open(source/"converted.jsonl.gz","rt") as f:
        verified = {r["seed"]:r["fights"] for r in map(json.loads,f)}
    total, exact, mask_errors, intent_matches = 0,0,0,0
    monster_offset,monster_count,monster_width=sts.OBS_LAYOUT["monsters"]
    intent_fields=[monster_offset+i*monster_width+k for i in range(monster_count) for k in (7,8,9)]
    fields = Counter()
    acts=Counter()
    examples = []
    with gzip.open(source/"replays.jsonl.gz","rt") as stream:
        for run in map(json.loads,stream):
            seed = run["seed"]
            if seed not in indices or seed not in verified:
                continue
            starts = indices[seed]
            if args.broad:
                candidates=[]
                rng=np.random.default_rng(seed)
                for act in sorted({f[0] for f in verified[seed]}):
                    ix=np.flatnonzero(starts["obs"][:,4]==act)
                    if len(ix) and acts[act]<256:
                        candidates.extend(rng.choice(ix,min(12,len(ix)),replace=False).tolist())
            else:
                candidates=range(min(5,len(starts)))
            for i in candidates:
                row=starts[i]
                step = int(row["step"])
                moves, _, n, _, _, _, _ = engine.convert_public_trace(seed,run["trace"][:step])
                if n!=step or not any(f[5]<=len(moves)<f[6] for f in verified[seed]):
                    continue
                env = sts.Env()
                env.reset(seed,20,final_act=True,character="ironclad")
                for a in moves:
                    if not env.step(a):
                        raise RuntimeError("translated prefix is illegal")
                native_obs=np.asarray(env.observe())
                x = projected(native_obs[None],rich)[0]
                reversed_obs=native_obs.copy()
                offset,n,width=sts.OBS_LAYOUT["draw"]
                pile=reversed_obs[offset:offset+n*width].reshape(n,width)
                pile[:]=pile[::-1].copy()
                if not np.array_equal(x, projected(reversed_obs[None],rich)[0]):
                    raise ValueError("public features depend on draw order")
                intent_matches+=int(np.array_equal(x[intent_fields],row["obs"][intent_fields]))
                diff = np.flatnonzero(x!=row["obs"])
                if len(diff):
                    fields.update(diff.tolist())
                    if len(examples)<12:
                        examples.append(dict(seed=seed,step=step,fields=[(int(i),int(x[i]),int(row["obs"][i])) for i in diff[:8]]))
                exact += not len(diff)
                mask_errors += not np.array_equal(np.asarray(env.mask())[:81],row["legal"])
                total += 1
                acts[int(row["obs"][4])]+=1
            if not args.broad and total>=512:
                break
    result = dict(decisions=total,identical=exact,mask_disagreements=mask_errors,
                  draw_reversal_checks=total, intent_fields_identical=intent_matches,
                  different_fields=fields.most_common(35),examples=examples)
    result["by_act"]=dict(acts)
    write(b/("broad-feature-audit.json" if args.broad else "feature-audit.json"),result)
    print(result,flush=True)


if __name__=="__main__":
    main()
