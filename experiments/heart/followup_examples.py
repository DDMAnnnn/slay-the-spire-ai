"""Reconstruct the displayed options at mandatory combat decisions from verified source games."""
from collections import Counter
import gzip
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from public_demo_export import digest
from public_followup import menu


def main():
    root=Path("build/public-expert-traces");out=Path("build/public-followups");out.mkdir(exist_ok=True,parents=True)
    sys.path[:0]=[str((root/"agent"/k).resolve()) for k in ("native","runtime")]
    import slaythespire as sts
    from sts_rl_env import SpireEnv
    stats=Counter();unsupported=Counter()
    tmp=out/"menus.tmp"
    with gzip.open(root/"replays.jsonl.gz","rt") as fp,gzip.open(tmp,"wt") as dst:
        for r in map(json.loads,fp):
            if r["error"] or not r["public_information"] or not r["original_audit"]["passed"]:continue
            env=SpireEnv(sts,sts.CharacterClass.IRONCLAD,r["seed"],20,"snapshot")
            if digest(env.snapshot())!=r["start_state_hash"]:raise ValueError("wrong initial state")
            saved=[]
            for domain,bits in r["trace"]:
                if domain==1 and bits>>29 in (2,3,5,6):
                    state=env.snapshot();legal=[int(a.bits) for a in env._native.legal_actions()]
                    if bits not in legal:raise ValueError("recorded follow-up was unavailable")
                    if len(legal)>1:
                        shown=menu(state,legal)
                        if shown is None:
                            unsupported[str((state["combat"]["card_select_task"],bits>>29))]+=1
                        else:
                            task,ctx,options=shown
                            distinct=sorted(set(options.values()))
                            if len(distinct)>1:
                                saved.append([r["seed"],task,ctx,distinct,distinct.index(options[bits]),
                                              distinct.index(options[min(legal)])])
                                stats[task]+=1
                env._native.step(sts.EnvironmentAction(sts.ActionDomain(domain),bits))
            if digest(env.snapshot())!=r["terminal_state_hash"]:raise ValueError("wrong final state")
            for item in saved:dst.write(json.dumps(item,separators=(",",":"))+"\n")
    tmp.replace(out/"menus.jsonl.gz")
    (out/"collection.json").write_text(json.dumps(dict(supported=stats,unsupported=unsupported),indent=2))
    print(stats,unsupported,flush=True)


if __name__=="__main__":main()
