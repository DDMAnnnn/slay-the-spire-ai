"""Collect visit counts and relative evaluations of alternative plays from public search particles."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from public_demo_export import combat_index, digest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--shard",type=int,required=True)
    p.add_argument("--shards",type=int,default=4);p.add_argument("--out",type=Path,default=Path("build/public-root-compatible"))
    args=p.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    root=Path("build/public-expert-traces")
    sys.path[:0]=[str((root/"agent"/k).resolve()) for k in ("native","runtime")]
    import slaythespire as sts
    from sts_rl_env import SpireEnv
    dest=out/f"roots-{args.shard}.jsonl.gz";tmp=dest.with_suffix(".tmp")
    n=0;games=0
    selection=random.Random(247341+args.shard)
    particles=random.Random(247631+args.shard)
    with gzip.open(root/"replays.jsonl.gz","rt") as src,gzip.open(tmp,"wt") as dst:
        for i,r in enumerate(map(json.loads,src)):
            if i%args.shards!=args.shard or r["error"] or not r["public_information"] or not r["original_audit"]["passed"]:continue
            seed=r["seed"];env=SpireEnv(sts,sts.CharacterClass.IRONCLAD,seed,20,"snapshot")
            if digest(env.snapshot())!=r["start_state_hash"]:raise ValueError("initial hash")
            saved=[]
            for step,(domain,bits) in enumerate(r["trace"]):
                if domain==1 and combat_index(bits) is not None:
                    legal={a.bits for a in env._native.legal_actions() if int(a.domain)==1}
                    if 0x80000000 in legal and len(legal)>2 and all(combat_index(a) is not None for a in legal):
                        act=env.snapshot()["act"]
                        if selection.random() < (.13,.28,.65,1.)[act-1]:
                            budget=160 if act<3 else 384
                            z=env.public_combat_search_action(budget,4,particles.randrange(1<<55))
                            best=combat_index(z["action"].bits)
                            estimates=[]
                            for a,nv,q in zip(z["root_actions"],z["root_visits"],z["root_mean_evaluations"]):
                                ix=combat_index(a.bits)
                                if ix is not None and a.bits in legal and nv>0 and math.isfinite(q):
                                    estimates.append([ix,int(nv),round(float(q),5)])
                            if best is not None and len(estimates)>=2:
                                saved.append([seed,step,act,best,estimates])
                env._native.step(sts.EnvironmentAction(sts.ActionDomain(domain),bits))
            if digest(env.snapshot())!=r["terminal_state_hash"]:raise ValueError("terminal hash")
            for row in saved:dst.write(json.dumps(row,separators=(",",":"))+"\n")
            n+=len(saved);games+=1
            if games%50==0:print(args.shard,games,n,flush=True)
    tmp.replace(dest);print("finished",args.shard,games,n,flush=True)


if __name__=="__main__":
    main()
