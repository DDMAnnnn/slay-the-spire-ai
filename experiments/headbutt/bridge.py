"""A20 Ironclad combat decisions from the pinned fork's public observation and mask."""
from dataclasses import dataclass
from functools import lru_cache

import sts_env as sts


def block(observation, name):
    start, count, width = sts.OBS_LAYOUT[name]
    return observation[start:start + count * width].reshape(count, width)


@lru_cache(maxsize=None)
def card_facts(raw_id, upgraded):
    return dict(zip(sts.CARD_FACT_NAMES, sts.CARD_FACTS[upgraded, raw_id]))


def card(row):
    raw_id, upgraded = int(row[0]) - 1, int(row[1])
    facts = card_facts(raw_id, upgraded)
    result = {"id": sts.CARD_IDS[raw_id], "upgrades": upgraded,
              "type": next((k.upper() for k in ("attack", "skill", "power", "curse", "status") if facts[k]), "")}
    if len(row) > 2:
        result["cost"] = int(row[2])
    elif facts["x_cost"] or facts["cost"] > 0:
        # The static table encodes both unplayable and free cards as zero.
        result["cost"] = -1 if facts["x_cost"] else int(facts["cost"])
    for key in ("damage", "block", "magic"):
        if facts[key]:
            result["base_" + key] = int(facts[key])
    if facts["hits"]:
        result["hits"] = int(facts["hits"])
    return result


def powers(values, names):
    return [{"id": name.replace("_", " ").title(), "amount": int(v)}
            for name, v in zip(names, values) if v]


@dataclass
class Decision:
    state: dict
    actions: dict  # policy action -> native action index, in mask order


def read(env):
    if sts.OBS_VERSION != 3:
        raise ValueError(f"Unsupported fork observation version {sts.OBS_VERSION}")
    obs = env.observe()
    g = block(obs, "global")[0]
    cls = block(obs, "class")[0]
    if not g[7] or sts.CHARACTERS[int(cls[0]) - 1] != "IRONCLAD":
        raise ValueError("This adapter expects an Ironclad combat decision")
    p = block(obs, "player")[0]
    normal = int(p[7]) == 1
    headbutt = int(p[7]) == 2 and int(g[26]) == 10
    if not (normal or headbutt):
        raise ValueError(f"Unsupported combat prompt: input {int(p[7])}, task {int(g[26])}")
    player = dict(current_hp=int(p[0]), max_hp=int(p[1]), block=int(p[2]),
                  gold=int(g[2]), floor=int(g[3]), ascension=int(g[5]),
                  powers=powers(p[sts.N_PLAYER_BASE:], sts.PLAYER_STATUS_NAMES),
                  relics=[{"id": sts.RELIC_NAMES[int(r[0]) - 1], **({"counter": int(r[1]) - 1} if r[1] else {})}
                          for r in block(obs, "relics") if r[0]],
                  potions=[sts.POTION_NAMES[int(v) - 1] if v else "Potion Slot"
                           for v in g[17:17 + int(g[16])]])
    player["class"] = "IRONCLAD"
    state = {"energy": int(p[3]), "turn": int(p[4]), "player": player, "monsters": []}
    for zone in ("hand", "draw", "discard", "exhaust"):
        state[zone] = [card(r) for r in block(obs, zone) if r[0]]
    for r in block(obs, "monsters"):
        if not r[0]:
            continue
        intent = sts.INTENT_NAMES[int(r[7]) - 1] if r[7] else "hidden"
        monster = dict(id=sts.MONSTER_NAMES[int(r[0]) - 1], current_hp=max(0, int(r[1])),
                       max_hp=int(r[2]), block=int(r[3]), targetable=bool(r[6]), intent=intent,
                       powers=powers(r[sts.MONSTER_BASE:], sts.MONSTER_STATUS_NAMES))
        if intent.startswith("ATTACK"):
            monster.update(damage=int(r[8]), hits=int(r[9]))
        state["monsters"].append(monster)

    actions = {}
    hand = block(obs, "hand")
    choices = block(obs, "choices")
    if headbutt:
        state["selection"] = "headbutt"
    for native in map(int, env.mask().nonzero()[0]):
        if normal and sts.ACT_PLAY <= native < sts.ACT_POTION:
            slot, target = divmod(native - sts.ACT_PLAY, 5)
            option = f"play {slot} {target}" if hand[slot, 4] else f"play {slot}"
        elif normal and native == sts.ACT_END_TURN:
            option = "end"
        elif headbutt and sts.ACT_CHOOSE <= native < sts.ACT_CONFIRM:
            slot = native - sts.ACT_CHOOSE
            kind, encoded, upgraded, _, _, legal = choices[slot]
            discard = state["discard"][slot]
            if (not legal or sts.CHOICE_KINDS[int(kind)] != "CARD" or
                    (sts.CARD_IDS[int(encoded) - 1], int(upgraded)) != (discard["id"], discard["upgrades"])):
                raise ValueError(f"Headbutt choice {slot} disagrees with the visible discard pile")
            option = f"retrieve {slot}"
        else:
            raise ValueError(f"Unsupported legal fork action {native} in this Headbutt integration")
        if option in actions:
            raise ValueError(f"Ambiguous native mapping for {option}")
        actions[option] = native
    if not actions:
        raise ValueError("No legal combat decision")
    return Decision(state, actions)
