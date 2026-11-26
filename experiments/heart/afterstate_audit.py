"""Perturb the true draw order and compare the critic's public branch estimates."""
from collections import Counter
import gzip
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from combat_afterstate import choose, features
from public_demo_export import combat_index, public_row


def main():
    root = Path("build/public-expert-traces")
    sys.path[:0] = [str((root / "agent" / name).resolve()) for name in ("native", "runtime")]
    import slaythespire as sts
    from sts_rl_env import SpireEnv
    out = Path("build/afterstate-critic")
    model = json.loads((out / "critic.json").read_text())
    layout = json.loads(Path("build/public-expert-intent/layout.json").read_text())
    layout.update(critic_cards=model["cards"],critic_enemies=model["enemies"])
    pairs = Counter()
    changed = attempts = 0
    with gzip.open(root / "replays.jsonl.gz", "rt", encoding="utf8") as fp:
        for game in map(json.loads,fp):
            if game["error"] or not game["public_information"] or not game["original_audit"]["passed"]:
                continue
            env = SpireEnv(sts,sts.CharacterClass.IRONCLAD,game["seed"],20,"snapshot")
            for d,b in game["trace"]:
                if d == 1 and pairs[game["seed"]] < 4:
                    s = env.snapshot()
                    legal = {a.bits for a in env._native.legal_actions()}
                    if 0x80000000 in legal and len(legal) >= 2 and len(s["combat"]["draw_pile"]) >= 2:
                        attempts += 1
                        alt = env.clone_with_resampled_hidden(278771 + attempts)
                        shadow = alt.snapshot()
                        first = [c[0] for c in s["combat"]["draw_pile"]]
                        second = [c[0] for c in shadow["combat"]["draw_pile"]]
                        if first != second:
                            if sorted(first) != sorted(second):
                                raise ValueError("different draw-pile contents")
                            vals = []
                            for branch, state in ((env,s),(alt,shadow)):
                                acts = {a.bits for a in branch._native.legal_actions()}
                                idx = {combat_index(bits) for bits in acts} - {None}
                                x = features(public_row(state,idx,layout),layout)
                                seed = 59103 + attempts
                                result = branch._native.public_belief_combat_search_action(384,4,seed,2,True)
                                scores = {}
                                move, compared = choose(branch,sts,state,result,seed,model,layout,1.5,3,scores)
                                vals.append((acts,x,move,compared,scores))
                            if vals[0] != vals[1]:
                                raise ValueError(("hidden-order sensitivity",game["seed"],attempts))
                            if vals[0][3]:
                                pairs[game["seed"]] += 1
                                changed += int(vals[0][2] != result.action.bits)
                env._native.step(sts.EnvironmentAction(sts.ActionDomain(d),b))
                if pairs[game["seed"]] >= 4:
                    break
            if sum(pairs.values()) >= 48:
                break
    n = sum(pairs.values())
    if n < 48:
        raise ValueError("insufficient genuinely reordered comparisons")
    report = dict(compared=n,games=len(pairs),replaced_search_move=changed,attempts=attempts)
    (out / "privacy.json").write_text(json.dumps(report,indent=2),encoding="utf8")
    print(report,flush=True)


if __name__ == "__main__":
    main()
