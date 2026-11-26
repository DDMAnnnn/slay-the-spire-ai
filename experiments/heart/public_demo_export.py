"""Replay the upstream agent with its native Windows CPython 3.13 environment; export public combat rows."""
import argparse
from array import array
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import struct
import sys


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def combat_index(bits):
    kind, source, target = bits >> 29, bits & 65535, (bits >> 16) & 8191
    if kind == 0 and source < 10 and target < 5:
        return 5 * source + target
    if kind == 1 and source < 5:
        if target == 8191:
            return 75 + source
        if target < 5:
            return 50 + 5 * source + target
    return 80 if kind == 4 else None


def public_row(state, legal, layout):
    # These are the only source fields the learner can see. Pile order is discarded in all three zones.
    o = [0] * layout["size"]
    c = state["combat"]
    hp, maximum = c["player_hp"], c["player_max_hp"]
    o[0], o[1], o[3], o[4], o[5], o[7] = hp, maximum, state["floor"], state["act"], 20, 1
    o[12:15] = state["keys"]
    o[16] = state["potion_capacity"]
    for i, potion in enumerate(c["potions"][:5]):
        o[17+i] = potion + 1 if potion > 1 else 0
    o[22], o[23] = len(state["deck"]), len(state["relics"])
    zones = layout["zones"]
    for name, cards in (("deck", state["deck"]), ("draw", c["draw_pile"]),
                        ("discard", c["discard_pile"]), ("exhaust", c["exhaust_pile"])):
        off, limit, width = zones[name]
        cards = sorted((card[0]+1, int(bool(card[1]))) for card in cards)
        for i, (card, up) in enumerate(cards[:limit]):
            o[off+i*width:off+i*width+2] = [card, up]
    off, limit, width = zones["relics"]
    for i, relic in enumerate(sorted(x[0]+1 for x in state["relics"])[:limit]):
        o[off+i*width] = relic
    off, _, _ = zones["player"]
    o[off:off+7] = [hp, maximum, c["player_block"], c["energy"], c["turn"], len(c["hand"]), c["draw_pile_size"]]
    for status, amount, _ in c["player_statuses"]:
        o[off+8+status] = amount
    dome = any(r[0] == layout["dome"] for r in state["relics"])
    off, limit, width = zones["monsters"]
    for i, m in enumerate(c["monsters"][:limit]):
        pos = off+i*width
        o[pos:pos+4] = [m[0]+1,*m[1:4]]
        o[pos+6] = m[4]
        if "rendering" in layout:
            from public_demo_intents import render_monster
            o[pos+7:pos+10] = render_monster(i,state,dome,layout["rendering"])
        else:
            o[pos+9] = m[11] if m[10] > 0 and not dome else 0
        for col, status in enumerate(("STRENGTH","VULNERABLE","WEAK","POISON"),6):
            o[pos+10+layout["monster_status"][status]] = m[col]
    off, limit, width = zones["hand"]
    for i, card in enumerate(c["hand"][:limit]):
        pos = off+i*width
        o[pos:pos+5] = [card[0]+1, bool(card[1]), card[2], any(a//5 == i for a in legal if a<50), card[10]]
    return array("h", map(int, o))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("build/public-expert-traces"))
    parser.add_argument("--source-root",type=Path)
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args()
    root = args.root
    source = args.source_root or root
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    sys.path[:0] = [str((source/"agent"/directory).resolve()) for directory in ("native", "runtime")]
    import slaythespire as sts
    from sts_rl_env import SpireEnv
    layout = json.loads((root/"layout.json").read_text())
    counts, by_act = Counter(), Counter()
    files = {}
    try:
        with gzip.open(source/"replays.jsonl.gz", "rt", encoding="utf8") as stream:
            for index,line in enumerate(stream):
                if index >= args.limit:
                    break
                r = json.loads(line)
                if r["error"] or not r["public_information"] or not r["original_audit"]["passed"]:
                    continue
                seed = int(r["seed"])
                bucket = int.from_bytes(hashlib.sha256(str(seed).encode()).digest()[:4], "big") % 10
                split = "train" if bucket<8 else "validation" if bucket==8 else "test"
                env = SpireEnv(sts, sts.CharacterClass.IRONCLAD, seed, 20, "snapshot")
                native = env._native
                state = env.snapshot()
                if digest(state) != r["start_state_hash"]:
                    raise ValueError(f"source initial hash mismatch for {seed}")
                game = bytearray()
                for step, (domain, bits) in enumerate(r["trace"]):
                    acts = {(int(a.domain), int(a.bits)): a for a in native.legal_actions()}
                    if (domain,bits) not in acts:
                        raise ValueError(f"unavailable demonstration action on {seed}")
                    if domain == 1 and (1, 0x80000000) in acts:
                        legal = {combat_index(b) for d,b in acts if d==1}
                        target = combat_index(bits)
                        if None not in legal and target in legal and len(legal)>1:
                            obs = public_row(state, legal, layout)
                            mask = bytes(int(i in legal) for i in range(81))
                            game.extend(struct.pack("<QI",seed,step) + obs.tobytes() + mask + bytes([target]))
                            counts[split] += 1
                            by_act[state["act"]] += 1
                    transition = native.step(acts[(domain,bits)])
                    state = env.snapshot()
                if digest(state) != r["terminal_state_hash"] or bool(transition.terminated) != r["terminated"]:
                    raise ValueError(f"source terminal hash mismatch for {seed}")
                # Only publish training rows after the complete trace passed the source hash.
                if split not in files:
                    files[split] = (root/f"obs-{split}.bin.tmp").open("wb")
                files[split].write(game)
                counts["games_verified"] += 1
                if (index+1)%100 == 0:
                    print(index+1,dict(counts),flush=True)
    finally:
        for f in files.values():
            f.close()
    for split in files:
        (root/f"obs-{split}.bin.tmp").replace(root/f"obs-{split}.bin")
    (root/"source-export.json").write_text(json.dumps(dict(counts=counts, by_act=by_act, limit=args.limit),indent=2))
    print(dict(counts),dict(by_act),flush=True)


if __name__ == "__main__":
    main()
