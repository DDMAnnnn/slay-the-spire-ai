"""Extract combat labels from verified prefixes of a public-information agent's published games."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import urllib.request

import heart_native as engine

from .train import write

REV = "1e69c2e78a5ab3cb44cca5ffdfcb82803a68701d"
URL = f"https://raw.githubusercontent.com/SnyderConsulting/sts_ml/{REV}/results/replays.jsonl.gz"
DIGEST = "6e7078b3e4fd09d77f373d619993a3783ae0820facc5ac3166fc32d0a7941789"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("build/public-expert-traces"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source = args.out / "replays.jsonl.gz"
    if not source.exists():
        with urllib.request.urlopen(URL, timeout=45) as response:
            source.write_bytes(response.read())
    if hashlib.sha256(source.read_bytes()).hexdigest() != DIGEST:
        raise ValueError("archive checksum mismatch")
    counts, stopped, acts = Counter(), Counter(), Counter()
    examples = []
    with gzip.open(source, "rt", encoding="utf8") as src, gzip.open(args.out / "converted.jsonl.gz", "wt", encoding="utf8") as out:
        for run in map(json.loads, src):
            counts["supplied"] += 1
            if run["error"] or not run["public_information"] or not (run.get("original_audit") or {}).get("passed"):
                counts["excluded_source"] += 1
                continue
            counts["source_fights"] += len(run["fights"])
            seed = int(run["seed"])
            actions, fights, consumed, reason, screen, floor, won = engine.convert_public_trace(seed, run["trace"])
            ranges = []
            for ours, theirs in zip(fights, run["fights"]):
                expected = [theirs[k] for k in ("act","floor","encounter","entry_hp","exit_hp")]
                if ours[:5] != expected:
                    reason = "fight_mismatch"
                    if len(examples) < 10:
                        examples.append(dict(seed=seed, actual=ours[:5], expected=expected))
                    break
                ranges.append(ours)
                acts[ours[0]] += 1
                counts["combat_decisions"] += ours[6] - ours[5]
            stopped[reason] += 1
            if reason == "end" and consumed == len(run["trace"]) and len(ranges) == len(run["fights"]) and won == run["terminal_victory"] and floor == run["maximum_floor"]:
                counts["complete_matching_games"] += 1
                counts["complete_matching_victories"] += int(won)
            if not ranges:
                continue
            counts["games_with_verified_fights"] += 1
            counts["verified_fights"] += len(ranges)
            # A game belongs to exactly one split, including all of its late encounters.
            bucket = int.from_bytes(hashlib.sha256(str(seed).encode()).digest()[:4], "big") % 10
            split = "test" if bucket == 9 else "validation" if bucket == 8 else "train"
            out.write(json.dumps(dict(seed=seed, split=split, fights=ranges,
                                      actions=actions[:ranges[-1][6]]), separators=(",",":"))+"\n")
    report = dict(source=URL, sha256=DIGEST, counts=dict(counts), endings=dict(stopped),
                  verified_fights_by_act=dict(acts), divergence_examples=examples)
    write(args.out / "conversion.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
