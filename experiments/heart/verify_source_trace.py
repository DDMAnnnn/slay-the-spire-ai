"""Replay an ordinary source run and verify every legal action and its terminal hash."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from public_demo_export import digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record",type=Path)
    args=parser.parse_args()
    data=json.loads(args.record.read_text(encoding="utf8"))
    root=Path("build/public-expert-traces")
    sys.path[:0]=[str((root/"agent"/k).resolve()) for k in ("native","runtime")]
    import slaythespire as sts
    from sts_rl_env import SpireEnv
    text=Path("build/rl-fork/include/constants/MonsterIds.h").read_text(encoding="utf8")
    names=re.findall(r'"([^"]+)"',text.split("monsterIdStrings[]",1)[1].split("};",1)[0])
    heart=names.index("CORRUPT_HEART")
    constants=Path("build/rl-fork/include/constants")
    def names_from(file,key):
        block=(constants/file).read_text(encoding="utf8").split(key,1)[1].split("};",1)[0]
        return re.findall(r'"([^"]*)"',block)
    cards=names_from("Cards.h","cardEnumStrings[]")
    relics=names_from("Relics.h","relicIds[]")
    env=SpireEnv(sts,sts.CharacterClass.IRONCLAD,int(data["seed"]),20,"snapshot")
    seen=False;keys={};first=None
    for i,(domain,bits) in enumerate(data["trace"]):
        state=env.snapshot()
        is_heart=(state["act"]==4 and state["in_combat"] and any(m[0]==heart for m in state["combat"]["monsters"]))
        if is_heart and not seen:
            entry_deck=Counter(cards[c[0]]+("+" if c[1] else "") for c in state["deck"])
            first=dict(step=i,floor=state["floor"],hp=state["combat"]["player_hp"],max_hp=state["max_hp"],
                       deck=dict(sorted(entry_deck.items())),relics=[relics[r[0]] for r in state["relics"]])
        seen|=bool(is_heart)
        for k,v in zip(("Ruby","Emerald","Sapphire"),state["keys"]):
            if v and k not in keys:keys[k]=state["floor"]
        legal={(int(a.domain),a.bits):a for a in env._native.legal_actions()}
        env._native.step(legal[domain,bits])
    end=env.snapshot()
    won=bool(seen and end["outcome"]==2)
    if not env._native.is_terminal or digest(end)!=data["terminal_hash"] or seen!=data["heart"] or won!=data["won"]:
        raise ValueError("The fresh simulator did not reproduce the stored outcome")
    if won and (len(keys)!=3 or end["hp"]<=0):
        raise ValueError("A winning replay should have all three keys and positive HP")
    result=dict(seed=str(data["seed"]),actions=len(data["trace"]),heart_entry=first,won=won,
                exit_hp=end["hp"],keys_by_floor=keys,terminal_hash=data["terminal_hash"])
    args.record.with_name("verification.json").write_text(json.dumps(result,indent=2),encoding="utf8")
    print(result,flush=True)


if __name__=="__main__":
    main()
