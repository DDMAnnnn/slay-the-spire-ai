"""Render only the icon and damage a player sees for an Ironclad enemy intent."""
import re
from pathlib import Path


def move_names(text):
    text = re.sub(r"//[^\n]*|/\*[\s\S]*?\*/","",text)
    match = re.search(r"enum class MonsterMoveId\s*:[^{]+\{([^}]*)",text)
    return [name.split("=")[0].strip() for name in match[1].split(",") if name.strip()]


def rendering_schema(source_moves):
    import sts_env as sts

    base = Path("build/rl-fork/include/constants")
    moves = move_names((base/"MonsterMoves.h").read_text())
    rows = (base/"MonsterIntents.h").read_text().split("monsterMoveIntent[] = {",1)[1].split("};",1)[0]
    table = re.findall(r"Intent::([A-Z_]+)",re.sub(r"//[^\n]*","",rows))
    if len(moves)!=len(table):
        raise ValueError("mismatched move and displayed intent table")
    icons = {name:i+1 for i,name in enumerate(sts.INTENT_NAMES)}
    icon = {name:icons[t] for name,t in zip(moves,table)}
    names = move_names(source_moves.read_text())
    return dict(moves=names,icon=[icon[k] for k in names],icons=icons,
                statuses={k:sts.PLAYER_STATUS_NAMES.index(k) for k in ("VULNERABLE","INTANGIBLE","SURROUNDED")},
                asleep=sts.MONSTER_STATUS_NAMES.index("ASLEEP"),
                odd=sts.RELIC_NAMES.index("Odd Mushroom"),krane=sts.RELIC_NAMES.index("Paper Krane"))


def render_monster(index, state, dome, schema):
    c=state["combat"]
    m=c["monsters"][index]
    if m[1]<=0 or m[12]:
        return 0,0,0
    icons=schema["icons"]
    if dome:
        return icons["NONE"],0,0
    name=schema["moves"][m[5]]
    icon=schema["icon"][m[5]]
    if name in ("DARKLING_HARDEN","DECA_SQUARE_OF_PROTECTION"):
        icon=icons["DEFEND_BUFF"]
    if name=="LAGAVULIN_SLEEP":
        asleep=any(k==schema["asleep"] and amount for k,amount in c["monster_statuses"][index])
        icon=icons["SLEEP"] if asleep else icons["STUN"]
    if icon > icons["ATTACK_DEFEND"]:
        return icon,0,0
    powers={k for k,_,_ in c["player_statuses"]}
    rels={k for k,_ in state["relics"]}
    status=schema["statuses"]
    damage=m[10]+m[6]
    if m[8]>0:
        damage*=.6 if schema["krane"] in rels else .75
    if status["VULNERABLE"] in powers:
        damage*=1.25 if schema["odd"] in rels else 1.5
    if c["stance"]==2:
        damage*=2
    if status["SURROUNDED"] in powers:
        last=c["last_targeted_monster"]
        if last!=index and 0<=last<len(c["monsters"]) and c["monsters"][last][1]>0:
            damage=int(damage*1.5)
    if status["INTANGIBLE"] in powers:
        damage=min(damage,1)
    return icon,max(0,int(damage)),m[11]
