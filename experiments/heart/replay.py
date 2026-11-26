"""Replay a frozen A20 policy from Neow and display its public decisions offline."""
import argparse
import gzip
import json
from pathlib import Path

import heart_native as engine
import numpy as np
import sts_env as sts

ROOT = Path(__file__).parent
REFERENCE = ROOT / "results"
VARIANTS = {"first": (40150942, False), "synergies": (41105607, True)}


def set_combat_preferences(preferences):
    engine.set_play_base(preferences.get("play_base", engine.initial_play_base()))
    if "drink_table" in preferences:
        engine.set_drinks(preferences["drink_table"], preferences["margin"])
    else:
        engine.reset_drinks()


def set_actor(params):
    if params is None:
        engine.reset_actor()
    else:
        engine.set_actor(*params)


def set_strategic_apprentice(config):
    if config:
        engine.set_strategic_apprentice(*config[:7])
        acts=config[7] if len(config)>7 else 15
        if hasattr(engine,"set_strategic_acts"):engine.set_strategic_acts(acts)
        elif acts!=15:raise ValueError("native build cannot restrict the strategic policy by act")
    elif hasattr(engine,"reset_strategic_apprentice"):
        engine.reset_strategic_apprentice()


def set_human_events(config):
    if config:
        engine.set_human_events(*config)
    elif hasattr(engine,"reset_human_events"):
        engine.reset_human_events()


def set_route_worlds(config):
    if config or hasattr(engine,"set_route_worlds"):
        engine.set_route_worlds(*(config or (0,4,3,0.)))


def set_learned_map(config):
    if config:
        engine.set_learned_map(*config[:6])
        if len(config)>6 or hasattr(engine,"set_map_acts"):
            engine.set_map_acts(config[6] if len(config)>6 else 7)
    elif hasattr(engine,"reset_learned_map"):
        engine.reset_learned_map()


def set_deck_forecast(config):
    if config:
        engine.set_deck_forecast(*config)
    elif hasattr(engine,"reset_deck_forecast"):
        engine.reset_deck_forecast()


def set_resource_policy(config):
    if config:
        engine.set_resource_policy(*config)
    elif hasattr(engine,"reset_resource_policy"):
        engine.reset_resource_policy()


def set_historical_gate(config):
    if config or hasattr(engine,"set_historical_gate"):
        engine.set_historical_gate(config if config is not None else [0.]*12)


def set_historical_menu(config):
    if config:
        engine.set_historical_menu(*config)
    elif hasattr(engine,"reset_historical_menu"):
        engine.reset_historical_menu()


def set_inventory_search(config):
    if config:
        engine.set_inventory_search(*config)
    elif hasattr(engine,"set_inventory_search"):
        engine.set_inventory_search(0,0,0.)


def set_public_belief(config,weights=None):
    if hasattr(engine,"set_terminal_search"):
        engine.set_terminal_search(*(config.get("terminal_search",[0,0.]) if config else [0,0.]))
    if hasattr(engine,"set_stasis_reconstruction"):
        engine.set_stasis_reconstruction(bool(config and config.get("stasis")))
    if hasattr(engine,"set_public_tree"):
        engine.set_public_tree(config.get("tree_budget",0) if config else 0, config.get("tree_depth",1) if config else 1)
    if hasattr(engine,"set_continuation"):
        engine.set_continuation(config.get("next_width",0) if config else 0, bool(config and config.get("free_plays")))
    if hasattr(engine,"set_belief_action_limit"):
        engine.set_belief_action_limit(config.get("action_limit",0) if config else 0)
    if hasattr(engine,"set_belief_plans"):
        engine.set_belief_plans(1,32.,False)
    if config:
        w=config.get("bank_weights",weights)
        if w is None: raise ValueError("restoring combat particles requires the collection weights")
        set_inventory_search(None)
        set_historical_menu(None)
        engine.collect_beliefs(w,config["seed"],config["count"])
        if config.get("enrich",0):
            engine.enrich_late_particles(w,config["enrich"])
        if "plans" in config:
            engine.set_belief_plans(config["plans"],config["win_value"],bool(config.get("enrich")))
        engine.set_belief(config["draws"],config["scope"],config["scale"])
    elif hasattr(engine,"set_belief"):
        engine.set_belief(0,0,0.)


def set_choice_attention(config):
    if config:
        engine.set_choice_attention(*config)
    else:
        engine.reset_choice_attention()


def set_inventory_value(config):
    if config:
        engine.set_inventory_value(*config)
    else:
        engine.reset_inventory_value()


def set_curriculum(config):
    if config:
        engine.set_curriculum(*config[:6])
        engine.set_curriculum_scope(config[6] if len(config)>6 else 0)
    else:
        engine.reset_curriculum()


def set_expert(config):
    if config:
        engine.set_expert(*config)
    else:
        engine.reset_expert()


def set_teacher(config):
    engine.set_teacher_scale(config[1] if config else 0.)
    if config:
        set_actor(config[0])


def set_late_combat(head):
    if head is None:
        engine.reset_late_combat()
    else:
        engine.set_late_combat(*head)


def set_final_weights(weights):
    if weights is None:
        engine.reset_final_weights()
    else:
        engine.set_final_weights(weights)


def set_opening(table):
    if table is None:
        engine.reset_neow()
    else:
        engine.set_neow(table)


def set_draft(table):
    if table is None:
        engine.reset_draft()
    else:
        engine.set_draft(table)


def load_model(variant, checkpoint=REFERENCE/"checkpoint.json.gz"):
    with gzip.open(checkpoint,"rt",encoding="utf8") as f:
        model=json.load(f)
    return apply_model(variant, model)


def set_forecast(params):
    if params is None:
        engine.reset_forecast()
    else:
        engine.set_forecast(params["intents"], params["settings"])


def set_boss_tactics(params):
    if params is None:
        engine.reset_boss_tactics()
    else:
        engine.set_boss_tactics(params)


def apply_model(variant, model):
    set_opening_search(model.get("opening_search"))
    set_route(model.get("route"))
    set_edits(model.get("deck_edits"))
    set_recovery(model.get("recovery"))
    set_forecast(model.get("forecast"))
    set_boss_tactics(model.get("boss_tactics"))
    engine.set_event_table(model["events"])
    engine.set_choice_table(model["choices"])
    engine.set_head(*model["combat"])
    engine.set_late(model["late"]["act"],model["late"]["weights"])
    engine.set_synergies(model.get("policy_synergies", VARIANTS[variant][1]))
    engine.set_first_boss_bonus(model.get("first_boss_relics", [0.] * engine.relic_count))
    engine.set_turn_depth(model.get("turn_depth", 2))
    set_combat_preferences(model.get("combat_preferences", {}))
    set_opening(model.get("neow"))
    set_draft(model.get("draft_context"))
    set_late_combat(model.get("late_combat"))
    engine.set_late_combat_last_act(model.get("late_combat_last_act",4))
    set_final_weights(model.get("final_weights"))
    set_actor(model.get("direct_actor"))
    set_teacher(model.get("trajectory_teacher"))
    set_expert(model.get("expert_prior"))
    set_choice_attention(model.get("choice_attention"))
    set_curriculum(model.get("curriculum_actor"))
    set_inventory_value(model.get("inventory_value"))
    engine.set_boss_draft(model.get("boss_draft",[]))
    set_scenarios(model.get("scenarios"))
    set_resource_policy(None)
    set_deck_forecast(None)
    set_learned_map(None)
    set_route_worlds(None)
    set_human_events(None)
    set_strategic_apprentice(None)
    if hasattr(engine,"reset_draft_plans"):
        engine.reset_draft_plans()
    if hasattr(engine,"reset_room_planner"):
        engine.reset_room_planner()
    if hasattr(engine,"reset_encounter_forecast"):
        engine.reset_encounter_forecast()
    if hasattr(engine,"reset_resource_effects"):
        engine.reset_resource_effects()
    set_public_belief(model.get("public_belief"),model["weights"])
    set_inventory_search(model.get("inventory_search"))
    set_historical_menu(model.get("historical_menu"))
    set_historical_gate(model.get("historical_gate"))
    set_resource_policy(model.get("resource_policy"))
    set_deck_forecast(model.get("deck_outcome"))
    set_learned_map(model.get("map_policy"))
    set_route_worlds(model.get("route_worlds"))
    set_human_events(model.get("human_events"))
    set_strategic_apprentice(model.get("strategic_apprentice"))
    plans=model.get("draft_plans")
    if plans:
        engine.set_draft_plans(plans["gate"],plans["prior"],plans["item"],plans["scales"])
    planner=model.get("room_planner")
    if planner:
        engine.set_room_planner(planner["transition"],planner["value"],planner["horizon"],planner["acts"],planner["gain"])
    encounter=model.get("encounter_forecast")
    if encounter:
        banks=[encounter["encounters"][str(a)] for a in (1,2,3)]
        engine.set_encounter_forecast(*encounter["weights"],banks,encounter["gain"],encounter["boss"])
    effects=model.get("resource_effects")
    if effects:
        engine.set_resource_effects(effects["hp"],effects["potion"],effects["gain"],effects["mix"],effects["typical"])
    return model["weights"]


def set_scenarios(params):
    if params is None:
        engine.reset_scenarios()
    else:
        engine.set_scenarios(*params)


def set_route(params):
    if params is None:
        engine.reset_route()
    else:
        engine.set_route(*params)


def set_opening_search(params):
    if params is None:
        engine.reset_opening_search()
    else:
        engine.set_opening_search(*params)


def set_recovery(params):
    if params is None:
        engine.reset_recovery()
    else:
        engine.set_recovery(*params)


def set_edits(table):
    if table is None:
        engine.reset_edits()
    else:
        engine.set_edits(table)


def rows(obs, key):
    start,count,width=sts.OBS_LAYOUT[key]
    return obs[start:start+count*width].reshape(count,width)


def card_name(row):
    return sts.CARD_NAMES[int(row[0])-1]+("+" if row[1] else "")


def label(obs,a):
    if a<sts.ACT_POTION:
        hand=rows(obs,"hand")[a//5]
        label=card_name(hand)
        if hand[4]:
            label+=" → "+sts.MONSTER_NAMES[int(rows(obs,"monsters")[a%5,0])-1]
        return label
    if a<sts.ACT_DISCARD:
        i=(a-sts.ACT_POTION)//5
        return "Drink "+sts.POTION_NAMES[int(obs[17+i])-1]
    if a<sts.ACT_END_TURN:
        return "Discard potion "+str(a-sts.ACT_DISCARD+1)
    if a==sts.ACT_END_TURN: return "End turn"
    if a==sts.ACT_CONFIRM: return "Confirm selection"
    index=a-sts.ACT_CHOOSE
    c=rows(obs,"choices")[index]
    kind=sts.CHOICE_KINDS[int(c[0])]
    if kind=="CARD": return card_name(c[1:])
    if kind=="RELIC": return sts.RELIC_NAMES[int(c[1])-1]
    if kind=="POTION": return sts.POTION_NAMES[int(c[1])-1]
    if kind=="GOLD": return f"Take {int(c[1])} gold"
    if kind=="KEY": return "Emerald key" if c[1]==1 else "Sapphire key"
    if kind=="MAP_NODE": return f"{sts.ROOM_NAMES[int(c[2])-1]} · column {index+1}"
    if kind=="REST_OPTION": return ("Rest","Smith","Recall","Lift","Toke","Dig","Leave")[index]
    if kind in ("EVENT_OPTION","NEOW_OPTION","MATCH_CARD"): return f"{kind.replace('_',' ').title()} {index+1}"
    return kind.replace("_"," ").title()


def heart_encounters(weights, start, panel):
    """Count actual combat entries, including a subsequent loss or truncation."""
    found = []
    for index in np.flatnonzero(panel[:, 1] == 4):
        seed = start + int(index)
        env = sts.Env()
        env.reset(seed, 20, final_act=True, character="ironclad")
        for action in engine.trace(weights, seed):
            obs = env.observe()
            if obs[7] and obs[4] == 4 and any(
                sts.MONSTER_NAMES[int(m[0])-1] == "CORRUPT_HEART"
                for m in rows(obs, "monsters") if m[6]
            ):
                found.append(seed)
                break
            if not env.step(action):
                raise RuntimeError("Heart encounter replay diverged")
    return found


def replay(variant,seed,checkpoint=None):
    w=load_model(variant) if checkpoint is None else load_model(variant,checkpoint)
    expected=REFERENCE/f"{seed}.json"
    reference=json.loads(expected.read_text()) if checkpoint is None and expected.exists() else None
    env=sts.Env()
    env.reset(seed,20,final_act=True,character="ironclad")
    frames=[]
    keys=[]
    previous=[False]*3
    while not env.done and len(frames)<3500:
        obs,mask=env.observe(),env.mask()
        scores=np.asarray(engine.scores(w,obs,mask))
        a=int(scores.argmax())
        if reference and (len(frames)>=len(reference["actions"]) or a!=reference["actions"][len(frames)]):
            raise AssertionError(f"Replay diverged at decision {len(frames)+1}")
        available=sorted(np.flatnonzero(mask),key=lambda i:-scores[i])[:5]
        p=rows(obs,"player")[0]
        frame=dict(act=int(obs[4]),floor=int(obs[3]),screen=sts.SCREEN_NAMES[int(obs[6])],
                   hp=int(obs[0]),max_hp=int(obs[1]),gold=int(obs[2]),action=label(obs,a),
                   keys=[bool(v) for v in obs[12:15]],
                   options=[[label(obs,i),round(float(scores[i]),2)] for i in available])
        if obs[7]:
            frame.update(turn=int(p[4])+1,block=int(p[2]),energy=int(p[3]),
                         hand=[card_name(h) for h in rows(obs,"hand") if h[0]],
                         enemies=[f"{sts.MONSTER_NAMES[int(m[0])-1]} · {int(m[1])}/{int(m[2])} HP · {int(m[3])} block"+
                                  (f" · attacking {int(m[8])} × {int(m[9])}" if m[9] else "")
                                  for m in rows(obs,"monsters") if m[6]])
        for i,(old,new) in enumerate(zip(previous,frame["keys"])):
            if not old and new: keys.append((("Ruby","Emerald","Sapphire")[i],int(obs[3])))
        previous=frame["keys"]
        frames.append(frame)
        if not mask[a] or not env.step(a): raise AssertionError("Illegal action")
    episode=env.episode()
    if reference:
        final=episode["path"][-1]
        assert env.done and env.won and all(previous) and not episode["truncated"]
        assert episode["character"]=="IRONCLAD" and episode["act"]==4 and episode["ascension"]==20
        assert final["encounter"]=="The Heart" and final["hp_after"]>0
        assert episode["actions"]==reference["actions"] and episode["path"]==reference["path"]
    return episode,frames,keys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant",choices=VARIANTS,default="first")
    parser.add_argument("--seed",type=int)
    parser.add_argument("--checkpoint",type=Path)
    parser.add_argument("--out",type=Path,default=Path("build/heart-replay.html"))
    args=parser.parse_args()
    seed=VARIANTS[args.variant][0] if args.seed is None else args.seed
    episode,frames,keys=replay(args.variant,seed,args.checkpoint)
    template=(ROOT/"replay.html").read_text(encoding="utf8")
    info=dict(seed=seed,won=episode["won"] and episode["act"]==4,hp=episode["hp"],
              max_hp=episode["max_hp"],frames=frames)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(template.replace("/*REPLAY_DATA*/null",json.dumps(info).replace("<","\\u003c")),encoding="utf8")
    print(f"Seed {seed}, {len(frames)} decisions, keys {keys}, final HP {episode['hp']}. {args.out}")


if __name__=="__main__":
    main()
