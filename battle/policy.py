"""One decision at a time: public objects -> contextual rows -> joint action scores."""
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import math
import re

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

from .reader import BASE_POTENCY, card_catalog, clauses, tokenize_description


# Each field has a value and a presence bit. A hidden intent differs from zero damage.
SCALARS = ("current_hp", "max_hp", "block", "energy", "turn", "gold", "ascension", "floor",
           "cost", "upgrades", "damage", "magic", "hits", "counter", "slot", "passive",
           "evoke", "D", "B", "M", "base_potency", "potency", "hand_count", "draw_count",
           "discard_count", "exhaust_count", "enemy_count", "potion_slots", "targetable")
ZONES = ("hand", "draw", "discard", "exhaust")
KINDS = ("play", "potion", "end")


@dataclass
class ObjectRow:
    role: str
    words: list = field(default_factory=list)
    numbers: dict = field(default_factory=dict)
    predicted_clauses: list = field(default_factory=list)


def observation_rows(state, reader, use_catalog=True, card_input="catalog"):
    rows, addresses = [], {}
    player = state["player"]
    direct = card_input in ("id", "id_text")
    use_catalog = use_catalog and not direct

    def add(role, obj, address=None):
        row = ObjectRow(role, [role], dict(obj))
        if address is not None:
            addresses[address] = len(rows)
        rows.append(row)
        return row

    def describe(row, kind, obj):
        if direct and kind == "card":
            # One identity token even when the ID itself contains English words or digits.
            identifier = hashlib.blake2s(obj["id"].encode(), digest_size=8).hexdigest()
            row.words.extend([kind, "id_" + identifier])
            if card_input == "id_text":
                description = obj.get("description") or ""
                for marker, field_name in (("!D!", "damage"), ("!B!", "block"), ("!M!", "magic")):
                    if field_name in obj:
                        description = description.replace(marker, str(obj[field_name]))
                row.words.append(description)
                for key, definition in sorted(obj.get("glossary", {}).items()):
                    row.words.extend([key, definition])
            return
        if not use_catalog:
            row.words.extend([kind, obj.get("description") or obj.get("id", "")])
            for keyword, definition in sorted(obj.get("glossary", {}).items()):
                row.words.extend([keyword, definition])
            return
        meaning = reader.interpret(kind, obj)
        row.words.extend([kind, meaning.source, meaning.program])
        if meaning.source == "predicted":
            for clause in clauses(obj["description"]):
                row.predicted_clauses.append((kind, tokenize_description(clause)[0]))
        if meaning.source == "unresolved":
            row.words.append(obj.get("id", "unknown"))
            for key, definition in sorted(obj.get("glossary", {}).items()):
                row.words.extend([key, definition])

    def powers(row, creature):
        # Keep each power and its stack count in its owner's row.
        for power in sorted(creature.get("powers", []), key=lambda p: p["id"]):
            describe(row, "power", power)
            row.words.extend(["stacks", str(power["amount"])])

    actor = add("player", player)
    actor.words.extend([player.get("class", ""), player.get("stance", "")])
    actor.numbers.update(energy=state["energy"], turn=state["turn"],
                         enemy_count=len(state["monsters"]), potion_slots=len(player["potions"]))
    powers(actor, player)
    for zone in ZONES:
        cards = state[zone]
        actor.numbers[zone + "_count"] = len(cards)
        for slot, card in enumerate(cards):
            row = add(zone, card, (zone, slot))
            row.words.extend([card.get("type", ""), card.get("target", "")])
            describe(row, "card", card)
            if direct:
                # Cost and upgrade count remain public in both arms. A second
                # numeric damage channel would reveal part of the withheld text.
                for field_name in ("damage", "block", "magic", "D", "B", "M"):
                    row.numbers.pop(field_name, None)
                continue
            if use_catalog and card["id"] in card_catalog():
                defaults = card_catalog()[card["id"]][2]
                for name, pair in defaults.items():
                    row.numbers[name] = pair[int(bool(card.get("upgrades")))]
            # Defaults describe a fresh card. A bridge can also supply instance
            # base values separately from the live display fields.
            for symbol, key in (("D", "base_damage"), ("B", "base_block"), ("M", "base_magic")):
                if key in card:
                    row.numbers[symbol] = card[key]
    for slot, monster in enumerate(state["monsters"]):
        row = add("enemy", monster, ("enemy", slot))
        intent = monster.get("intent", "hidden")
        row.words.extend([monster["id"], intent])
        if intent.startswith("ATTACK"):
            row.words.append("On(EnemyActs,Repeat(Hits,Attack(Player,DisplayedDamage)))")
            row.numbers.setdefault("hits", 1)
        powers(row, monster)
    for slot, item in enumerate(player.get("potion_details", player["potions"])):
        obj = {"id": item} if isinstance(item, str) else item
        row = add("potion", obj, ("potion", slot))
        describe(row, "potion", obj)
        if obj["id"] in BASE_POTENCY:
            row.numbers["base_potency"] = BASE_POTENCY[obj["id"]]
    for relic in player.get("relics", []):
        describe(add("relic", relic), "relic", relic)
    for slot, orb in enumerate(player.get("orb_slots", [])):
        row = add("orb", orb)
        row.words.append(orb["id"])
        row.numbers["slot"] = slot
    return rows, addresses


@lru_cache(maxsize=8192)
def token_id(token):
    # Stable buckets let an unresolved mod identifier enter a saved model.
    digest = hashlib.blake2s(token.encode("utf8"), digest_size=4).digest()
    return int.from_bytes(digest, "little") % 8192


def lexical_tokens(row):
    # Keep delimiters and operand order. Repeat(2,Attack(...)) is one program.
    return re.findall(r"\w+|[^\w\s]", " | ".join(row.words).casefold())


def scalars(row):
    values = []
    for key in SCALARS:
        value = row.numbers.get(key)
        x = float(value) if value is not None else 0.
        values.extend((math.copysign(math.log1p(abs(x)), x), float(value is not None)))
    return values


def action_indices(option, addresses):
    parts = option.split()
    if parts == ["end"]:
        return 2, -1, -1
    if len(parts) not in (2, 3) or parts[0] not in ("play", "potion"):
        raise ValueError(f"Unsupported bridge action {option!r}")
    kind = parts[0]
    source = addresses[("hand" if kind == "play" else "potion", int(parts[1]))]
    target = addresses[("enemy", int(parts[2]))] if len(parts) == 3 else -1
    return KINDS.index(kind), source, target


class CombatPolicy(nn.Module):
    def __init__(self, width=64, heads=4, reader=None, card_input="catalog"):
        super().__init__()
        if card_input not in ("catalog", "id", "id_text"):
            raise ValueError(f"Unknown card input: {card_input}")
        if reader is None and card_input == "catalog":
            from .reader_training import new_reader
            reader = new_reader()
        self.reader = reader
        self.card_input = card_input
        self.heads = heads
        self.memory_runtime = None
        self.tokens = nn.Embedding(8192, width)
        self.literal = nn.Linear(1, width, bias=False)
        self.program = nn.GRU(width, width, batch_first=True)
        self.numeric = nn.Linear(2 * len(SCALARS), width)
        self.belief = nn.Linear(len(reader.templates), width, bias=False) if reader is not None else None
        self.input_norm = nn.LayerNorm(width)
        layer = nn.TransformerEncoderLayer(width, heads, 2 * width, dropout=0.,
                                           batch_first=True, activation="gelu")
        self.battle = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.kind = nn.Embedding(len(KINDS), 16)
        self.absent = nn.Parameter(torch.zeros(1, width))
        self.action = nn.Sequential(nn.Linear(3 * width + 16, 2 * width), nn.GELU())
        self.score = nn.Linear(2 * width, 1)
        self.outcomes = nn.Linear(2 * width, 3)

    def embed_rows(self, rows):
        device = self.tokens.weight.device
        tokens = [lexical_tokens(r) for r in rows]
        sequences = [torch.tensor([token_id("#number" if t.isdecimal() else t) for t in row],
                                  device=device) for row in tokens]
        magnitudes = [self.tokens.weight.new_tensor([math.log1p(int(t)) if t.isdecimal() else 0.
                                                     for t in row]) for row in tokens]
        embedded = (self.tokens(pad_sequence(sequences, batch_first=True)) +
                    self.literal(pad_sequence(magnitudes, batch_first=True).unsqueeze(-1)))
        packed = pack_padded_sequence(embedded,
                                      [len(s) for s in sequences], batch_first=True, enforce_sorted=False)
        _, hidden = self.program(packed)
        numeric = self.tokens.weight.new_tensor([scalars(r) for r in rows])
        if self.belief is None:
            return self.input_norm(hidden[0] + self.numeric(numeric))
        # Keep the discrete program inspectable. The uncertainty over candidate
        # rules also gives outcome prediction a gradient path into the reader.
        beliefs = []
        for row in rows:
            guesses = [self.reader.linear(self.reader.vector(k, w)).softmax(0)
                       for k, w in row.predicted_clauses]
            beliefs.append(torch.stack(guesses).mean(0) if guesses
                           else numeric.new_zeros(len(self.reader.templates)))
        return self.input_norm(hidden[0] + self.numeric(numeric) + self.belief(torch.stack(beliefs)))

    def encode(self, rows):
        return self.battle(self.embed_rows(rows).unsqueeze(0)).squeeze(0)

    def read_state(self, state):
        return observation_rows(state, self.reader, use_catalog=self.reader is not None and
                                self.memory_runtime is None, card_input=self.card_input)

    def score_rows(self, players, sources, targets, kinds):
        fused = torch.cat([players, sources, targets, self.kind(kinds)], dim=-1)
        actions = self.action(fused)
        return self.score(actions).squeeze(-1), self.outcomes(actions)

    def forward(self, state, options):
        rows, addresses = self.read_state(state)
        context = self.encode(rows)
        indices = [action_indices(option, addresses) for option in options]
        kind, src, tgt = torch.tensor(indices, device=context.device).T
        lookup = torch.cat([context, self.absent])
        sources = lookup[src]
        if self.memory_runtime is not None:
            from .memory import cue_for, source
            interpreter, bank = self.memory_runtime
            meanings = []
            for action in options:
                played = source(state, action)
                if played:
                    kind_name, obj, target = played
                    _, meaning = interpreter.interpret(cue_for(kind_name, obj, state, target), bank)
                    meanings.append(meaning.to(context))
                else:
                    meanings.append(context.new_zeros(context.shape[-1]))
            sources = sources + torch.stack(meanings)
        return self.score_rows(context[0].expand(len(options), -1), sources, lookup[tgt], kind)

    @torch.no_grad()
    def choose(self, state, options):
        logits, _ = self(state, options)
        return options[logits.argmax().item()]
