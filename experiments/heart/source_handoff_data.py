"""Restore independently recorded Act 3/4 starts to measure downstream conditional performance."""
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

    pools = {3: [], 4: []}
    with gzip.open(root / "replays.jsonl.gz", "rt") as fp:
        for record in map(json.loads, fp):
            if record["error"] or not record["original_audit"]["passed"] or not record["public_information"]:
                continue
            if not record["act3"]:
                continue
            game = SpireEnv(sts, sts.CharacterClass.IRONCLAD, record["seed"], 20, "snapshot")
            if digest(game.snapshot()) != record["start_state_hash"]:
                raise ValueError("source initial state differs")
            seen = set()
            for i, (domain, bits) in enumerate(record["trace"], 1):
                game._native.step(sts.EnvironmentAction(sts.ActionDomain(domain), bits))
                obs = game.snapshot()
                act = obs["act"]
                if act in pools and act not in seen:
                    pools[act].append(dict(seed=record["seed"], prefix=record["trace"][:i],
                                           state_hash=digest(obs), act=act, floor=obs["floor"],
                                           reference_heart=record["heart_reached"], reference_win=record["heart_killed"]))
                    seen.add(act)
            if digest(game.snapshot()) != record["terminal_state_hash"]:
                raise ValueError("source terminal state differs")
    dest = Path("build/source-handoff")
    dest.mkdir(parents=True, exist_ok=True)
    for act, games in pools.items():
        if len({item["seed"] for item in games}) != len(games):
            raise ValueError("duplicate starting game")
        with gzip.open(dest / f"act-{act}.json.gz", "wt") as fp:
            json.dump(games, fp, separators=(",", ":"))
        print(act, len(games), sum(item["reference_heart"] for item in games), flush=True)


if __name__ == "__main__":
    main()
