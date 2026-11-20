"""An episodic interpreter. IDs address memories; the neural predictor sees public features."""
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import math
from pathlib import Path
import re

import torch
from torch import nn
from torch.nn import functional as F


ZONES = ("hand", "draw", "discard", "exhaust")
TEXT_DIM, CONTEXT_DIM, WIDTH = 96, 64, 64


@lru_cache(maxsize=8192)
def bucket(word, size):
    digest = hashlib.blake2s(word.encode(), digest_size=4).digest()
    return int.from_bytes(digest, "little") % size


def vector(items, size):
    result = [0.] * size
    for name, value in items:
        result[bucket(name, size)] += math.copysign(math.log1p(abs(value)), value)
    return result


def identity(kind, obj):
    return (kind, obj["id"], obj.get("upgrades", 0))


def text_features(kind, obj):
    text = (obj.get("description") or "").casefold()
    for marker, field in (("!d!", "damage"), ("!b!", "block"), ("!m!", "magic")):
        if obj.get(field, -1) >= 0:
            text = text.replace(marker, str(obj[field]))
    words = re.findall(r"[^\W\d_]+|\d+", text)
    tokens = [kind, obj.get("type", ""), *words, *(" ".join(p) for p in zip(words, words[1:]))]
    return vector(((w, 1) for w in tokens if w), TEXT_DIM)


def bag(cards):
    return Counter(f"{c['id']}@{c.get('upgrades', 0)}" for c in cards)


def visible(state, target):
    fields = {"energy": state["energy"], "turn": state["turn"]}
    player = state["player"]
    entities = [("player", player)]
    entities.extend((f"enemy[{i}]", monster) for i, monster in enumerate(state["monsters"]))
    if target is not None:
        entities.append(("target", state["monsters"][target]))
    for name, obj in entities:
        for k in ("current_hp", "block"):
            fields[f"{name}.{k}"] = obj[k]
        for p in obj.get("powers", []):
            fields[f"{name}.power.{p['id']}"] = p["amount"]
    for relic in player.get("relics", []):
        fields["relic/" + relic["id"]] = 1
        if "counter" in relic:
            fields["counter/" + relic["id"]] = relic["counter"]
    for i, item in enumerate(player.get("potions", [])):
        identifier = item if isinstance(item, str) else item["id"]
        fields[f"potion[{i}]/{identifier}"] = 1
    if player.get("stance"):
        fields["stance/" + player["stance"]] = 1
    for i, orb in enumerate(player.get("orb_slots", [])):
        fields[f"orb[{i}]/{orb['id']}"] = 1
    for zone in ZONES:
        for name, n in bag(state[zone]).items():
            fields[f"{zone}/{name}"] = n
    return fields


def context_features(state, target):
    fields = visible(state, target)
    if target is not None:
        fields["enemy/" + state["monsters"][target]["id"]] = 1
    for m in state["monsters"]:
        fields["present/" + m["id"]] = 1
    return vector(fields.items(), CONTEXT_DIM)


def source(state, action):
    parts = action.split()
    if parts[0] == "play":
        kind, obj = "card", state["hand"][int(parts[1])]
    elif parts[0] == "potion":
        kind = "potion"
        item = state["player"].get("potion_details", state["player"]["potions"])[int(parts[1])]
        obj = {"id": item} if isinstance(item, str) else item
    else:
        return None
    target = int(parts[2]) if len(parts) == 3 else None
    return kind, obj, target


@dataclass
class Cue:
    key: tuple
    text: list
    context: list


@dataclass
class Experience:
    cue: Cue
    changes: dict
    fight: str
    step: int


def cue_for(kind, obj, state, target=None):
    return Cue(identity(kind, obj), text_features(kind, obj), context_features(state, target))


def observe(state, action, after, fight, step):
    played = source(state, action)
    if not played or after is None or after.get("turn") != state["turn"]:
        return None
    kind, obj, target = played
    if target is not None and [m["id"] for m in state["monsters"]] != [m["id"] for m in after["monsters"]]:
        return None
    before, later = visible(state, target), visible(after, target)
    delta = {k: later.get(k, 0) - before.get(k, 0) for k in before.keys() | later.keys()}
    # Gross observed arrivals after removing the played copy. Unlike net hand size,
    # this also detects drawing another copy of the same card. It does not identify cause.
    remaining = bag(state["hand"])
    if kind == "card":
        remaining[f"{obj['id']}@{obj.get('upgrades', 0)}"] -= 1
    delta["hand.arrivals"] = sum((bag(after["hand"]) - remaining).values())
    return Experience(cue_for(kind, obj, state, target), {k:v for k,v in delta.items() if v}, str(fight), step)


class MemoryBank:
    def __init__(self, episodes=()):
        self.by_key = defaultdict(list)
        self.seen = set()
        for item in episodes:
            self.remember(item)

    def remember(self, item):
        token = (item.fight, item.step)
        if token not in self.seen:
            self.by_key[item.cue.key].append(item)
            self.seen.add(token)

    def candidates(self, cue, per_object=12, objects=6):
        # Exact identity first; lexical similarity selects other definitions.
        def relevance(key):
            text = self.by_key[key][-1].cue.text
            return (key == cue.key, sum(a*b for a,b in zip(cue.text, text)) /
                    max(1., math.sqrt(sum(v*v for v in text))))
        keys = sorted(self.by_key, key=relevance, reverse=True)[:objects]
        return [episode for key in keys for episode in self.by_key[key][-per_object:]]

    def episodes(self):
        return [episode for group in self.by_key.values() for episode in group]


def scale(field):
    return 8. if field.endswith((".current_hp", ".block")) else 1.


class MemoryInterpreter(nn.Module):
    """Cross-attend to observed outcomes and transfer residuals between contexts."""
    def __init__(self, fields, retrieval="grouped"):
        super().__init__()
        self.fields = tuple(fields)
        self.retrieval = retrieval
        self.embed = nn.Sequential(nn.Linear(TEXT_DIM + CONTEXT_DIM, WIDTH), nn.Tanh())
        self.query = nn.Linear(WIDTH, 24)
        self.key = nn.Linear(WIDTH, 24)
        self.prior = nn.Linear(WIDTH, len(fields))
        if retrieval == "pooled":
            self.identity_weight = nn.Parameter(torch.tensor(2.))

    def forward(self, inputs, memories, effects, match, valid):
        q, k = self.embed(inputs), self.embed(memories)
        scores = (F.normalize(self.query(q), dim=-1).unsqueeze(1) *
                  F.normalize(self.key(k), dim=-1)).sum(-1) * 6
        exact = valid & match.bool()
        related = valid & ~exact
        def attend(mask):
            return scores.masked_fill(~mask, -1e6).softmax(-1) * mask
        # Treat the entire related pool as one prior observation. Its influence
        # decays with direct evidence regardless of how many old cards we own.
        count = exact.sum(-1, keepdim=True)
        prior_count = related.any(-1, keepdim=True).to(scores)
        weights = (count * attend(exact) + prior_count * attend(related)) / (count + prior_count).clamp_min(1)
        if self.retrieval == "pooled":
            weights = (scores + self.identity_weight * match).masked_fill(~valid,-1e6).softmax(-1) * valid
        residual = ((effects - self.prior(k)) * weights.unsqueeze(-1)).sum(1)
        prediction = self.prior(q) + residual
        # Predictive embedding, in the same width as a combat object row.
        return prediction, q + F.linear(prediction, self.prior.weight.T), weights

    def tensors(self, cues, supports):
        B, K = len(cues), max(1, max(map(len, supports)))
        dev = self.prior.weight.device
        def tensor(data):
            return torch.tensor(data, device=dev, dtype=torch.float32)
        x = tensor([c.text+c.context for c in cues])
        keys = x.new_zeros(B,K,TEXT_DIM+CONTEXT_DIM)
        effects = x.new_zeros(B,K,len(self.fields))
        match = x.new_zeros(B,K)
        valid = torch.zeros(B,K,dtype=torch.bool,device=dev)
        for i, episodes in enumerate(supports):
            n = len(episodes)
            if n:
                keys[i,:n] = tensor([e.cue.text+e.cue.context for e in episodes])
                effects[i,:n] = tensor([self.labels(e) for e in episodes])
                match[i,:n] = tensor([e.cue.key == cues[i].key for e in episodes])
                valid[i,:n] = True
        return x, keys, effects, match, valid

    def labels(self, episode):
        return [episode.changes.get(k,0)/scale(k) for k in self.fields]

    @torch.no_grad()
    def interpret(self, cue, bank):
        episodes = bank.candidates(cue)
        prediction, embedding, weights = self(*self.tensors([cue],[episodes]))
        changes = {k:v * scale(k) for k,v in zip(self.fields, prediction[0].tolist())}
        # Newly observed fields can still be retrieved before any decoder retraining.
        for w,e in zip(weights[0].tolist(),episodes):
            for field, val in e.changes.items():
                if field not in self.fields:
                    changes[field] = changes.get(field,0.) + w*val
        extra = [(k,v) for k,v in changes.items() if k not in self.fields]
        return changes, embedding[0] + embedding.new_tensor(vector(extra, WIDTH))


def save_memory(path, model, bank):
    from tempfile import NamedTemporaryFile
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(version=2, retrieval=model.retrieval, fields=model.fields, weights=model.state_dict(),
                episodes=[asdict(e) for e in bank.episodes()])
    with NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        torch.save(data,tmp_path)
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)


def load_memory(path):
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data["version"] not in (1,2):
        raise ValueError("Unsupported memory version")
    model = MemoryInterpreter(data["fields"], data.get("retrieval","pooled"))
    model.load_state_dict(data["weights"])
    episodes = [Experience(Cue(**row["cue"]), row["changes"], row["fight"], row["step"])
                for row in data["episodes"]]
    return model.eval(), MemoryBank(episodes)
