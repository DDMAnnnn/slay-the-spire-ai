"""Test public reconstruction on naturally earned A20 Bronze Automaton fights."""
import argparse
from pathlib import Path

import heart_native as engine
import numpy as np

from .heart_curriculum import read_model
from .matched_fights import SETTINGS, configure
from .replay import apply_model
from .train import write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--threads",type=int,default=4)
    p.add_argument("--out",type=Path,default=Path("build/stasis-search"))
    args=p.parse_args()
    out=args.out
    out.mkdir(parents=True,exist_ok=True)
    parent=read_model("build/stratified-gate/challenger.json.gz")
    w=apply_model("first",parent)
    start=230000000
    path=out/"scout.npy"
    if not path.exists():
        np.save(path,np.asarray(engine.batch(w,start,32768,args.threads)))
    scout=np.load(path)
    available=np.flatnonzero(scout[:,0]>=32)
    seeds=start+np.random.default_rng(230113).choice(available,min(1536,len(available)),replace=False)
    engine.collect_beliefs(w,190000000,3072)
    engine.enrich_late_particles(w,192)
    audit=np.zeros(5,dtype=int)
    results={name:[] for name in ("original","reconstruct")}
    for i in range(0,len(seeds),32):
        file=out/f"paired-{i}.npz"
        if not file.exists():
            configure(SETTINGS["parent"])
            engine.set_stasis_reconstruction(False)
            bank=engine.Bank(w,seeds[i:i+32].tolist(),32,args.threads)
            diagnostic=bank.audit_stasis(w)
            arms={}
            for name in results:
                configure(SETTINGS["late_examples"])
                engine.set_stasis_reconstruction(name=="reconstruct")
                arms[name]=np.asarray(bank.entry_health(w,0,args.threads))
            np.savez(file,audit=diagnostic,**arms)
        with np.load(file) as data:
            audit+=data["audit"]
            for name in results:
                results[name].append(data[name])
    orig,new=(np.concatenate(x) for x in results.values())
    if not np.array_equal(orig[:,:5],new[:,:5]):
        raise ValueError("boss or entry conditions changed before play")
    # Resolve the encounter index from the fork's public name table.
    import sts_env as sts
    auto=list(sts.ENCOUNTER_NAMES).index("Automaton")
    is_auto=orig[:,0]==auto
    if is_auto.sum()!=audit[0] or not audit[2]:
        raise ValueError(f"unexpected encounter or no inferred cards: {is_auto.sum()}, {audit}")
    if not np.array_equal(orig[~is_auto],new[~is_auto]):
        raise ValueError("unrelated boss fight changed")
    def score(a):
        v=a[is_auto]
        return dict(wins=int(v[:,5].sum()),mean_exit_hp=float(v[:,6].mean()),capped=int(v[:,9].sum()))
    a,b=orig[is_auto,5]>0,new[is_auto,5]>0
    report=dict(start=start,ordinary_scout_count=len(scout),earned_floor32=len(available),sample=len(seeds),
                automaton_fights=int(is_auto.sum()),other_controls=int((~is_auto).sum()),
                stasis_decisions=int(audit[1]),inferred=int(audit[2]),correct_multisets=int(audit[3]),
                ambiguous_owner=int(audit[4]), original=score(orig),reconstruction=score(new),
                gained=int((b&~a).sum()),lost=int((a&~b).sum()),seeds=seeds[is_auto].tolist())
    write(out/"report.json",report)
    print({k:v for k,v in report.items() if k!="seeds"},flush=True)


if __name__=="__main__":
    main()
