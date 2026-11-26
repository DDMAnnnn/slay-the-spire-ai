"""Collect visible fight starts and observed results from fully reproduced reference games."""
import gzip
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from public_demo_export import digest


def main():
    root = Path("build/public-expert-traces")
    sys.path[:0] = [str((root / "agent" / name).resolve()) for name in ("native", "runtime")]
    import slaythespire as sts
    from sts_rl_env import SpireEnv
    out = Path("build/encounter-forecast")
    out.mkdir(exist_ok=True, parents=True)
    path = out / "fights.jsonl.gz"
    temp = path.with_suffix(".tmp")
    count = games = 0
    with gzip.open(root / "replays.jsonl.gz", "rt") as fp, gzip.open(temp, "wt") as dst:
        for r in map(json.loads, fp):
            if r["error"] or not r["public_information"] or not r["original_audit"]["passed"]:
                continue
            env = SpireEnv(sts, sts.CharacterClass.IRONCLAD, r["seed"], 20, "snapshot")
            if digest(env.snapshot()) != r["start_state_hash"]:
                raise ValueError("incorrect reset")
            starts = []
            prev = None
            for d,bits in r["trace"]:
                env._native.step(sts.EnvironmentAction(sts.ActionDomain(d), bits))
                s = env.snapshot()
                key = (s["act"], s["floor"], s["decision"]["encounter"]) if s.get("combat") else None
                if key and key != prev:
                    f = r["fights"][len(starts)]
                    if key != (f["act"], f["floor"], f["encounter"]):
                        raise ValueError("wrong encounter")
                    starts.append(dict(seed=r["seed"], index=len(starts), act=s["act"], floor=s["floor"],
                                       room=s["current_room"], hp=s["hp"], max_hp=s["max_hp"],
                                       deck=[(c[0], int(bool(c[1]))) for c in s["deck"]],
                                       relics=[v[0] for v in s["relics"]], potions=[v for v in s["potions"] if v>1],
                                       monsters=[m[0] for m in s["combat"]["monsters"]],
                                       encounter=f["encounter"], survived=f["survived"], exit_hp=f["exit_hp"]))
                prev=key
            if len(starts)!=len(r["fights"]) or digest(env.snapshot())!=r["terminal_state_hash"]:
                raise ValueError("source replay diverged")
            for item in starts:
                dst.write(json.dumps(item,separators=(",",":"))+"\n")
            games+=1;count+=len(starts)
    temp.replace(path)
    print(games,count,flush=True)


if __name__=="__main__":
    main()
