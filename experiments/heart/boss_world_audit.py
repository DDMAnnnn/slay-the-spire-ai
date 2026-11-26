"""Check planning leaves the real run intact and ignores its unrevealed future."""
import gzip
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from boss_relic_worlds import choose
from public_demo_export import digest


def router(native, state, legal):
    options = [a.bits for a in legal if int(a.domain) == 0 and a.bits < 0x80000000]
    if not options:
        return legal[0].bits
    if state["screen_state"] == 8:
        return next((a for a in options if (a >> 27) & 7 == 6), options[0])
    return min(options)


def main():
    root = Path("build/public-expert-traces")
    sys.path[:0] = [str((root / "agent" / p).resolve()) for p in ("native","runtime")]
    import slaythespire as sts
    from sts_rl_env import SpireEnv
    checked=[]
    with gzip.open(root/"replays.jsonl.gz","rt",encoding="utf8") as fp:
        for row in map(json.loads,fp):
            if row["error"] or not row["public_information"] or not row["original_audit"]["passed"]:
                continue
            if not any(f["act"] == 2 for f in row["fights"]):
                continue
            env=SpireEnv(sts,sts.CharacterClass.IRONCLAD,row["seed"],20,"snapshot")
            tested=False
            for d,bits in row["trace"]:
                s=env.snapshot()
                if not tested and s["screen_state"]==3 and s["act"]==1 and not s["in_combat"]:
                    legal=env._native.legal_actions()
                    if bits < 4 and len(legal)>1:
                        seen={r[0] for r in s["relics"]} | {r for r in s["decision"]["boss_relics"] if r>=0}
                        shadow=env.clone()
                        shadow._native=env._native.public_strategic_particle(81637,[sts.RelicId(i) for i in sorted(seen)])
                        a=choose(env,sts,s,legal,bits,router,seen,2)
                        b=choose(shadow,sts,shadow.snapshot(),shadow._native.legal_actions(),bits,router,seen,2)
                        if not a[1] or a!=b:
                            raise ValueError(("public branch values differ",row["seed"],a,b))
                        tested=True
                env._native.step(sts.EnvironmentAction(sts.ActionDomain(d),bits))
            if tested:
                if digest(env.snapshot()) != row["terminal_state_hash"]:
                    raise ValueError(("planning changed the recorded future",row["seed"]))
                checked.append(str(row["seed"]))
            if len(checked)==6:
                break
    if len(checked)!=6:
        raise ValueError("insufficient verified boss offers")
    out=Path("build/boss-worlds");out.mkdir(exist_ok=True,parents=True)
    (out/"privacy.json").write_text(json.dumps(dict(matched=checked,worlds=2,seed_source="public chest and inventory"),indent=2))
    print("Matched branch values and terminal hashes on",len(checked),"recorded games",flush=True)


if __name__ == "__main__":
    main()
