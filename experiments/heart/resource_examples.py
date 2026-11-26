"""Extract public camp, shop, smith and removal menus from verified benchmark play."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from public_demo_export import digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,default=Path("build/public-expert-traces"))
    p.add_argument("--out",type=Path,default=Path("build/resource-policy"))
    args=p.parse_args()
    args.out.mkdir(exist_ok=True,parents=True)
    sys.path[:0]=[str((args.source/"agent"/a).resolve()) for a in ("native","runtime")]
    import slaythespire as sts
    from sts_rl_env import SpireEnv
    sizes=json.loads((args.out/"sizes.json").read_text())
    cards,relics=sizes["cards"],sizes["relics"]
    strikes,defends=sizes["strikes"],sizes["defends"]
    def option(state,bits):
        screen=state["screen_state"]
        if bits & 0x80000000:
            return None
        index=bits & 255
        if screen==7:
            return (2+index,2,0,0)
        d=state["decision"]
        if screen==4:
            card,up,*_=d["selectable_cards"][index]
            return (9+card,3,int(bool(up)),0)
        kind=(bits>>27)&7
        if kind==0:
            card,up,_,price=d["shop_cards"][index]
            return (9+card,3,int(bool(up)),price)
        if kind==3:
            potion,price=d["shop_potions"][index]
            return (9+cards+relics+potion,5,0,price)
        if kind==4:
            relic,price=d["shop_relics"][index]
            return (9+cards+relic,4,0,price)
        if kind==5:
            return (1,1,0,d["shop_remove_cost"])
        if kind==6:
            return (0,0,0,0)
        raise ValueError(f"unsupported shop action {bits}")
    counts=Counter()
    dest_path=args.out/"menus.jsonl.gz"
    tmp=dest_path.with_suffix(".tmp.gz")
    with gzip.open(args.source/"replays.jsonl.gz","rt") as src,gzip.open(tmp,"wt") as dest:
        for r in map(json.loads,src):
            if r["error"] or not r["public_information"] or not r["original_audit"]["passed"]:
                continue
            seed=r["seed"]
            env=SpireEnv(sts,sts.CharacterClass.IRONCLAD,seed,20,"snapshot")
            state=env.snapshot()
            if digest(state)!=r["start_state_hash"]:
                raise ValueError("initial state mismatch")
            bucket=int.from_bytes(hashlib.sha256(str(seed).encode()).digest()[:4],"big")%10
            game=[]
            for domain,bits in r["trace"]:
                screen=state["screen_state"]
                task=state["decision"]["card_select_screen_type"]
                if domain==0 and (screen in (7,8) or screen==4 and task in (3,4)):
                    actions={a.bits:option(state,a.bits) for a in env._native.legal_actions()}
                    selected=actions[bits]
                    legal=sorted({x for x in actions.values() if x is not None})
                    if selected is not None and len(legal)>1:
                        deck=state["deck"]
                        inventory=sorted([9+x[0] for x in deck]+[9+cards+x[0] for x in state["relics"]])
                        base=[state["hp"],state["max_hp"],state["gold"],state["floor"],state["act"],
                              *state["keys"],len(deck),sum(bool(x[1]) for x in deck),
                              sum(x[0]==strikes for x in deck),sum(x[0]==defends for x in deck),
                              sum(x>1 for x in state["potions"]),state["potion_capacity"]]
                        menu=0 if screen==7 else 1 if screen==8 else 2 if task==3 else 3
                        game.append([seed,bucket,menu,base,inventory,legal,legal.index(selected)])
                        counts[menu]+=1
                action=sts.EnvironmentAction(sts.ActionDomain(domain),bits)
                env._native.step(action)
                state=env.snapshot()
            if digest(state)!=r["terminal_state_hash"]:
                raise ValueError("resource replay diverged")
            for row in game:
                dest.write(json.dumps(row,separators=(",",":"))+"\n")
    tmp.replace(dest_path)
    print("eligible menus by type",dict(counts),flush=True)


if __name__=="__main__":
    main()
