"""A20 Ironclad combat decisions from the pinned fork's public observation and mask."""
from battle.policy import Candidate, CombatDecision
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


# Pinned fork's CardSelectTask enum. Choice rows supply the actual objects.
# A zone of "offer" uses the grid's public order, including for a sorted draw grid.
PROMPTS = {
    1: ("hand", "Upgrade a card in hand"),
    2: ("offer", "Shuffle a chosen card into the draw pile"),
    3: ("offer", "Choose a card to add to hand"),
    4: ("hand", "Copy a card in hand"),
    5: ("hand", "Exhaust a card from hand"),
    6: ("hand", "Choose cards in hand to exhaust"),
    7: ("exhaust", "Return an exhausted card to hand"),
    8: ("hand", "Put a card on the bottom of the draw pile at zero cost until played"),
    9: ("hand", "Discard selected cards and draw replacements"),
    10: ("discard", "Headbutt choose discard to draw pile top"),
    11: ("discard", "Return a discarded card to hand"),
    12: ("discard", "Return a discarded card to hand at zero cost this turn"),
    13: ("discard", "Return selected cards to hand and retain them"),
    14: ("hand", "Choose a card to copy next turn"),
    15: ("hand", "Exhaust a card for energy"),
    16: ("offer", "Choose a skill from the draw pile for hand"),
    17: ("offer", "Choose an attack from the draw pile for hand"),
    18: ("offer", "Choose from the draw pile for hand"),
    19: ("hand", "Put a card on top of the draw pile at zero cost until played"),
    20: ("hand", "Put a card from hand on top of the draw pile"),
    21: ("hand", "Choose cards to discard"),
    22: ("hand", "Choose cards to retain"),
    23: ("offer", "Scry: choose cards from the top of the draw pile to discard"),
    24: ("offer", "Choose a draw pile card to play twice and exhaust"),
    25: ("offer", "Choose a Wish effect"),
    26: ("offer", "Choose a stance"),
    27: ("offer", "Choose an attack to add to hand"),
    28: ("hand", "Put selected cards at the bottom at zero cost until played"),
}
PLAY = "Play a combat action"
TARGET_POTIONS = {"Fear Potion", "Fire Potion", "Poison Potion", "Weak Potion"}


def event_decision(obs, mask):
    """Score the offered event buttons. Inventory management is a separate decision."""
    g = block(obs, "global")[0]
    if g[7] or sts.SCREEN_NAMES[int(g[6])] != "EVENT" or g[9] <= 0:
        raise ValueError("Expected a public event screen")
    player = dict(current_hp=int(g[0]), max_hp=int(g[1]), gold=int(g[2]),
                  floor=int(g[3]), ascension=int(g[5]),
                  relics=[{"id": sts.RELIC_NAMES[int(r[0]) - 1],
                           **({"counter": int(r[1]) - 1} if r[1] else {})}
                          for r in block(obs, "relics") if r[0]],
                  potions=[sts.POTION_NAMES[int(v) - 1] if v else "Potion Slot"
                           for v in g[17:17 + int(g[16])]])
    player["class"] = sts.CHARACTERS[int(block(obs, "class")[0, 0]) - 1]
    candidates = []
    for slot, row in enumerate(block(obs, "choices")):
        native = sts.ACT_CHOOSE + slot
        if mask[native]:
            if not row[5]:
                raise ValueError(f"Event button {slot} missing from public choices")
            candidates.append(Candidate(native, "select", ("event_option", slot)))
    if not candidates:
        raise ValueError("Event has no legal buttons")
    state = {"event_id": int(g[9]) - 1, "player": player,
             "deck": [card(r) for r in block(obs, "deck") if r[0]],
             "event_options": [c.source[1] for c in candidates]}
    return CombatDecision(state, "Choose an event option", candidates)


def read(env):
    if sts.OBS_VERSION != 3:
        raise ValueError(f"Unsupported fork observation version {sts.OBS_VERSION}")
    obs = env.observe()
    g = block(obs, "global")[0]
    cls = block(obs, "class")[0]
    if sts.CHARACTERS[int(cls[0]) - 1] != "IRONCLAD":
        raise ValueError("This adapter expects Ironclad")
    if not g[7] and sts.SCREEN_NAMES[int(g[6])] == "EVENT":
        return event_decision(obs, env.mask())
    if not g[7]:
        raise ValueError("Expected combat or an event screen")
    p = block(obs, "player")[0]
    normal = int(p[7]) == 1
    task = int(g[26])
    if normal:
        zone, prompt = None, PLAY
    elif int(p[7]) == 2 and task in PROMPTS:
        zone, prompt = PROMPTS[task]
        if task == 3 and not block(obs, "choices")[3, 5]:
            prompt += " at zero cost this turn"
    else:
        raise ValueError(f"Unsupported combat prompt: input {int(p[7])}, task {task}")
    player = dict(current_hp=int(p[0]), max_hp=int(p[1]), block=int(p[2]),
                  gold=int(g[2]), floor=int(g[3]), ascension=int(g[5]),
                  powers=powers(p[sts.N_PLAYER_BASE:], sts.PLAYER_STATUS_NAMES),
                  relics=[{"id": sts.RELIC_NAMES[int(r[0]) - 1], **({"counter": int(r[1]) - 1} if r[1] else {})}
                          for r in block(obs, "relics") if r[0]],
                  potions=[sts.POTION_NAMES[int(v) - 1] if v else "Potion Slot"
                           for v in g[17:17 + int(g[16])]])
    player["class"] = "IRONCLAD"
    state = {"energy": int(p[3]), "turn": int(p[4]), "player": player, "monsters": []}
    for pile in ("hand", "draw", "discard", "exhaust"):
        state[pile] = [card(r) for r in block(obs, pile) if r[0]]
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

    hand = block(obs, "hand")
    choices = block(obs, "choices")
    if zone == "offer":
        state["offer"] = []
    refs = {}
    if zone:
        for slot, (kind, encoded, up, picked, _, legal) in enumerate(choices):
            if not kind:
                continue
            name = sts.CHOICE_KINDS[int(kind)]
            if name == "SKIP":
                continue
            if name == "EVENT_OPTION" and task == 26:
                obj = {"id": sts.STANCE_NAMES[int(encoded) - 1]}
            elif name == "CARD":
                obj = card((encoded, up))
            else:
                raise ValueError(f"Unsupported {name} in {prompt}")
            if zone == "offer":
                ref = (zone, len(state[zone]))
                state[zone].append(obj)
            else:
                ref = (zone, slot)
                visible = state[zone][slot]
                if (visible["id"], visible["upgrades"]) != (obj["id"], obj["upgrades"]):
                    raise ValueError(f"Choice {slot} disagrees with {zone}")
            if picked:
                state[zone][ref[1]]["selected"] = True
            if legal:
                refs[slot] = ref
    candidates = []
    for native in map(int, env.mask().nonzero()[0]):
        if sts.ACT_PLAY <= native < sts.ACT_POTION:
            slot, t = divmod(native - sts.ACT_PLAY, 5)
            c = Candidate(native, "play", ("hand", slot), ("enemy", t) if hand[slot, 4] else None)
        elif native < sts.ACT_DISCARD:
            slot, t = divmod(native - sts.ACT_POTION, 5)
            targeted = player["potions"][slot] in TARGET_POTIONS
            c = Candidate(native, "potion", ("potion", slot), ("enemy", t) if targeted else None)
        elif native < sts.ACT_END_TURN:
            c = Candidate(native, "discard_potion", ("potion", native - sts.ACT_DISCARD))
        elif native == sts.ACT_END_TURN:
            c = Candidate(native, "end")
        elif native == sts.ACT_CONFIRM:
            c = Candidate(native, "confirm")
        else:
            slot = native - sts.ACT_CHOOSE
            if sts.CHOICE_KINDS[int(choices[slot, 0])] == "SKIP":
                c = Candidate(native, "skip")
            else:
                c = Candidate(native, "select", refs[slot])
        candidates.append(c)
    if not candidates:
        raise ValueError("No legal combat decision")
    return CombatDecision(state, prompt, candidates)
