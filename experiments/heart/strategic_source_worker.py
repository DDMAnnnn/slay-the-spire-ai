"""Play ordinary games with our strategic scorer and the reference public combat search."""
import argparse
from array import array
import base64
from collections import Counter
import json
from pathlib import Path
import random
import re
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from strategic_projection import StrategicProjection
from public_demo_export import combat_index, digest, public_row


def send(msg):
    print(json.dumps(msg,separators=(",",":")), flush=True)


def reference_budget(state, heart):
    floor = state["floor"]
    monsters = state["combat"]["monsters"]
    if floor in (50, 51) or (state["act"] == 4 and any(m[0] == heart for m in monsters)):
        return 50000
    if floor in (16, 17, 33, 34) or state["act"] == 4:
        return 20000
    return 2048 if len(monsters) >= 3 or sum(max(int(m[1]), 0) for m in monsters) >= 180 else 256


def main():
    p=argparse.ArgumentParser();p.add_argument("--early",type=int,required=True);p.add_argument("--late",type=int,required=True)
    p.add_argument("--mode",type=int,default=-1);p.add_argument("--elite",type=int,default=0)
    p.add_argument("--boss",type=int,default=0)
    p.add_argument("--reference-combat",action="store_true")
    p.add_argument("--search-followups",action="store_true")
    p.add_argument("--neural-combat",action="store_true")
    args=p.parse_args()
    root=Path("build/public-expert-traces");sys.path[:0]=[str((root/"agent"/k).resolve()) for k in ("native","runtime")]
    import slaythespire as sts
    from sts_rl_env import SpireEnv
    codec=StrategicProjection()
    layout=json.loads(Path("build/public-expert-intent/layout.json").read_text())
    text=Path("build/rl-fork/include/constants/MonsterIds.h").read_text(encoding="utf8").split("monsterIdStrings[]",1)[1].split("};",1)[0]
    heart=re.findall(r'"([^"]+)"',text).index("CORRUPT_HEART")
    for command in sys.stdin:
        req=json.loads(command)
        if req.get("stop"):return
        seed=req["seed"];env=SpireEnv(sts,sts.CharacterClass.IRONCLAD,seed,20,"snapshot")
        if "prefix" in req:
            for domain, bits in req["prefix"]:
                env._native.step(sts.EnvironmentAction(sts.ActionDomain(domain), bits))
            if digest(env.snapshot()) != req["state_hash"]:
                raise ValueError("wrong restored starting position")
        rng=random.Random(260375);menus=Counter();entered=False;searches=0
        for step in range(3500):
            state=env.snapshot();legal=env._native.legal_actions()
            if not legal:raise ValueError(("no actions",state["floor"],state["screen_state"]))
            if state["in_combat"] and state["act"]==4:
                entered |= any(m[0]==heart for m in state["combat"]["monsters"])
            if len(legal)==1:action=legal[0]
            elif state["in_combat"]:
                followup = all(a.bits != 0x80000000 for a in legal)
                if not followup or args.search_followups:
                    budget=args.late if state["act"]>=3 else args.early
                    if state["current_room"]==3:budget=max(budget,args.elite)
                    if state["current_room"]==6:budget=max(budget,args.boss)
                    particle_seed=rng.getrandbits(53)
                    if args.reference_combat:
                        budget = reference_budget(state, heart)
                        search = env._native.public_belief_combat_search_action(
                            budget, 8, env._native.public_combat_observation_hash(), 2, True)
                        bits = search.action.bits
                        if bits not in {a.bits for a in legal}:
                            raise ValueError(("illegal public search action", state["floor"], bits))
                        searches += 1
                        if followup:
                            menus["searched_followup"] += 1
                    elif budget==0:
                        mapping={combat_index(a.bits):a.bits for a in legal if combat_index(a.bits) is not None}
                        o=public_row(state,set(mapping),layout)
                        if not args.neural_combat:
                            o[2]=state["gold"];o[6]=state["screen_state"];o[8]=state["current_room"]+1
                            o[15]=state["act_boss"]+1;o[layout["zones"]["class"][0]]=state["character_class"]+1
                        packet=base64.b64encode(array("f",o).tobytes()).decode("ascii")
                        send({"obs":packet,"legal":list(mapping),"neural":args.neural_combat})
                        bits=mapping[json.loads(sys.stdin.readline())];menus["local_combat"]+=1
                    elif args.mode<0:
                        bits=env.public_combat_search_action(budget,4,particle_seed)["action"].bits
                        searches+=1
                    else:
                        bits=env._native.public_belief_combat_search_action(budget,4,particle_seed,args.mode,True).action.bits
                        searches+=1
                    action=sts.EnvironmentAction(sts.ActionDomain(1),bits)
                else:
                    action=min(legal,key=lambda a:int(a.bits));menus["combat_followup"]+=1
            elif state["screen_state"]==1 and state["current_event"]==31:
                # The source API exposes a pair choice. Sample legal positions without reading face-down cards.
                action=legal[rng.randrange(len(legal))];menus["match_pairs"]+=1
            else:
                burning=env._native.public_burning_elite_position()
                obs,keys=codec.encode(state,[a.bits for a in legal],burning)
                packet=base64.b64encode(array("f",obs).tobytes()).decode("ascii")
                send({"obs":packet,"legal":list(keys)})
                pick=json.loads(sys.stdin.readline())
                action=sts.EnvironmentAction(sts.ActionDomain(0),keys[pick])
                menus[str(state["screen_state"])]+=1
            move=env._native.step(action)
            if move.terminated:break
        end=env.snapshot()
        send({"result":{"seed":str(seed),"floor":end["floor"],"act":end["act"],"heart":bool(entered),
                        "won":bool(entered and end["outcome"]==2),"capped":not move.terminated,
                        "steps":step+1,"searches":searches,"menus":dict(menus)}})


if __name__=="__main__":
    main()
