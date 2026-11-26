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


def load_model(variant):
    with gzip.open(REFERENCE/"checkpoint.json.gz","rt",encoding="utf8") as f:
        model=json.load(f)
    engine.set_event_table(model["events"])
    engine.set_choice_table(model["choices"])
    engine.set_head(*model["combat"])
    engine.set_late(model["late"]["act"],model["late"]["weights"])
    engine.set_synergies(VARIANTS[variant][1])
    return model["weights"]


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
    if kind=="CARD": return card_name(c)
    if kind=="RELIC": return sts.RELIC_NAMES[int(c[1])-1]
    if kind=="POTION": return sts.POTION_NAMES[int(c[1])-1]
    if kind=="GOLD": return f"Take {int(c[1])} gold"
    if kind=="KEY": return "Emerald key" if c[1]==1 else "Sapphire key"
    if kind=="MAP_NODE": return f"{sts.ROOM_NAMES[int(c[2])-1]} · column {index+1}"
    if kind=="REST_OPTION": return ("Rest","Smith","Recall","Lift","Toke","Dig","Leave")[index]
    if kind in ("EVENT_OPTION","NEOW_OPTION","MATCH_CARD"): return f"{kind.replace('_',' ').title()} {index+1}"
    return kind.replace("_"," ").title()


def replay(variant,seed):
    w=load_model(variant)
    expected=REFERENCE/f"{seed}.json"
    reference=json.loads(expected.read_text()) if expected.exists() else None
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
    parser.add_argument("--out",type=Path,default=Path("build/heart-replay.html"))
    args=parser.parse_args()
    seed=VARIANTS[args.variant][0] if args.seed is None else args.seed
    episode,frames,keys=replay(args.variant,seed)
    template=(ROOT/"replay.html").read_text(encoding="utf8")
    info=dict(seed=seed,won=episode["won"] and episode["act"]==4,hp=episode["hp"],
              max_hp=episode["max_hp"],frames=frames)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(template.replace("/*REPLAY_DATA*/null",json.dumps(info).replace("<","\\u003c")),encoding="utf8")
    print(f"Seed {seed}, {len(frames)} decisions, keys {keys}, final HP {episode['hp']}. {args.out}")


if __name__=="__main__":
    main()
