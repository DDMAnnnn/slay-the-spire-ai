"""Vary concealed draws before Headbutt; compare public-search actions and root statistics."""
from collections import Counter
import gzip
import json
from pathlib import Path
import re
import sys


def main():
    root=Path("build/public-expert-traces");sys.path[:0]=[str((root/"agent"/p).resolve()) for p in ("native","runtime")]
    import slaythespire as sts
    from sts_rl_env import SpireEnv
    path=Path("build/rl-fork/include/constants/Cards.h")
    block=path.read_text(encoding="utf8").split("cardEnumStrings[]",1)[1].split("};",1)[0]
    headbutt=re.findall(r'"([^"]+)"',block).index("HEADBUTT")
    compared=0;games=Counter();failures=[];candidates=0
    with gzip.open(root/"replays.jsonl.gz","rt") as fp:
        for r in map(json.loads,fp):
            if r["error"] or not r["public_information"] or not r["original_audit"]["passed"]:continue
            env=SpireEnv(sts,sts.CharacterClass.IRONCLAD,r["seed"],20,"snapshot")
            for d,bits in r["trace"]:
                step=sts.EnvironmentAction(sts.ActionDomain(d),bits)
                shadow=None
                if d==1 and bits>>29==0 and games[r["seed"]]<4:
                    s=env.snapshot()["combat"];index=bits&0xffff
                    if s["hand"][index][0]==headbutt and len(s["draw_pile"])>=2:
                        candidates+=1
                        shadow=env.clone_with_resampled_hidden(63409+3713*candidates)
                        shadow._native.step(step)
                env._native.step(step)
                if shadow is None:continue
                current=env.snapshot().get("combat");alternative=shadow.snapshot().get("combat")
                if not current or not alternative or current["card_select_task"]!=10:continue
                legal=sorted(a.bits for a in env._native.legal_actions())
                if len(legal)<2 or any(k>>29!=2 for k in legal):continue
                if legal!=sorted(a.bits for a in shadow._native.legal_actions()):continue
                # This diagnostic alone looks at order to prove that the negative control changed it.
                left=[c[:3] for c in current["draw_pile"]];right=[c[:3] for c in alternative["draw_pile"]]
                if left==right or sorted(left)!=sorted(right):continue
                seed=env._native.public_combat_observation_hash()
                if seed!=shadow._native.public_combat_observation_hash():continue
                def search(world):
                    v=world._native.public_belief_combat_search_action(384,8,seed,2,True)
                    return v.action.bits,[(a.bits,n,round(q,10)) for a,n,q in zip(v.root_actions,v.root_visits,v.root_mean_evaluations)]
                a,b=search(env),search(shadow)
                compared+=1;games[r["seed"]]+=1
                if a!=b:failures.append(dict(seed=str(r["seed"]),decision=compared,action_changed=a[0]!=b[0]))
                if compared>=48:break
            if compared>=48:break
    result=dict(pairs=compared,distinct_games=len(games),predecessors_tried=candidates,failures=failures)
    (Path("build/public-followups")/"privacy.json").write_text(json.dumps(result,indent=2))
    if compared<40 or failures:raise ValueError(result)
    print(result,flush=True)


if __name__=="__main__":main()
