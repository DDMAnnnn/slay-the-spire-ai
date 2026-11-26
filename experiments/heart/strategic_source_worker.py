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
from public_followup import choose
from boss_relic_worlds import choose as plan_relic
from combat_afterstate import choose as compare_moves
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
    p.add_argument("--move-correction",type=Path)
    p.add_argument("--relic-worlds",type=int,default=0)
    p.add_argument("--independent-trees",action="store_true")
    p.add_argument("--reference-combat",action="store_true")
    p.add_argument("--followup-model",type=Path)
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
    followup_weights=json.loads(args.followup_model.read_text()) if args.followup_model else None
    critic=None
    if args.move_correction:
        critic=json.loads(Path("build/afterstate-critic/critic.json").read_text())
        critic["correction"]=json.loads(args.move_correction.read_text())
        layout.update(critic_cards=critic["cards"],critic_enemies=critic["enemies"])
    def router(native,state,legal):
        burning=native.public_burning_elite_position()
        obs,keys=codec.encode(state,[a.bits for a in legal],burning)
        packet=base64.b64encode(array("f",obs).tobytes()).decode("ascii")
        send({"obs":packet,"legal":list(keys)})
        return keys[json.loads(sys.stdin.readline())]

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
        fights=[]
        observed_relics=set();relic_comparison=None
        trace=[] if req.get("record") else None
        previous_combat=False
        for step in range(3500):
            state=env.snapshot();legal=env._native.legal_actions()
            if not legal:raise ValueError(("no actions",state["floor"],state["screen_state"]))
            if args.relic_worlds and not state["in_combat"]:
                screen=state["screen_state"];shown=state["decision"]
                observed_relics.update(r[0] for r in state["relics"])
                if screen==2:observed_relics.update(shown["relic_rewards"])
                if screen==3:observed_relics.update(shown["boss_relics"])
                if screen==8:observed_relics.update(r for r,price in shown["shop_relics"] if price>=0)
                observed_relics.discard(-1)
            in_combat=state["in_combat"]
            if in_combat and not previous_combat:
                combat=state["combat"]
                fights.append(dict(act=state["act"],floor=state["floor"],room=state["current_room"],
                                   hp=combat["player_hp"],max_hp=combat["player_max_hp"],
                                   enemies=[int(monster[0]) for monster in combat["monsters"]]))
            previous_combat=in_combat
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
                        if followup:menus["searched_followup"]+=1
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
                        search=env._native.public_belief_combat_search_action(budget,4,particle_seed,args.mode,not args.independent_trees)
                        bits=search.action.bits
                        if critic and budget==384 and not followup:
                            bits,checked=compare_moves(env,sts,state,search,particle_seed,critic,layout,1.,3)
                            menus["checked_move"]+=int(checked)
                            menus["corrected_move"]+=int(bits!=search.action.bits)
                        searches+=1
                    action=sts.EnvironmentAction(sts.ActionDomain(1),bits)
                else:
                    if followup_weights:
                        bits,learned=choose(state,[a.bits for a in legal],followup_weights)
                        action=sts.EnvironmentAction(sts.ActionDomain(1),bits)
                        menus["learned_followup" if learned else "unsupported_followup"]+=1
                    else:
                        action=min(legal,key=lambda a:int(a.bits));menus["combat_followup"]+=1
            elif state["screen_state"]==1 and state["current_event"]==31:
                # The source API exposes a pair choice. Sample legal positions without reading face-down cards.
                action=legal[rng.randrange(len(legal))];menus["match_pairs"]+=1
            else:
                bits=router(env._native,state,legal)
                if args.relic_worlds and state["act"]==1 and state["screen_state"]==3:
                    baseline=bits
                    bits,samples=plan_relic(env,sts,state,legal,bits,router,observed_relics,args.relic_worlds)
                    relic_comparison=dict(parent=baseline,selected=bits,samples=samples)
                action=sts.EnvironmentAction(sts.ActionDomain(0),bits)
                menus[str(state["screen_state"])]+=1
            if trace is not None:trace.append([int(action.domain),int(action.bits)])
            move=env._native.step(action)
            if move.terminated:break
        end=env.snapshot()
        send({"result":{"seed":str(seed),"floor":end["floor"],"act":end["act"],"heart":bool(entered),
                        "won":bool(entered and end["outcome"]==2),"capped":not move.terminated,
                        "outcome":end["outcome"],"exit_hp":end["hp"],"keys":list(end["keys"]),
                        "steps":step+1,"searches":searches,"menus":dict(menus),"fight_entries":fights,
                        **({"relic_comparison":relic_comparison} if relic_comparison is not None else {}),
                        **({"trace":trace,"terminal_hash":digest(end)} if trace is not None else {})}})


if __name__=="__main__":
    main()
