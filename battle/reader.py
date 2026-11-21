"""Public object descriptions -> ordered rule programs. No game simulation."""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re

import torch
from torch import nn


@dataclass(frozen=True)
class Meaning:
    program: str
    source: str


@lru_cache(maxsize=1)
def card_catalog():
    table = Path(__file__).resolve().parents[1] / "CARD_TABLE.md"
    catalog = {}
    for line in table.read_text(encoding="utf8").splitlines():
        fields = [f.strip() for f in line.strip("|").split("|")]
        if len(fields) != 6 or not fields[1].startswith(chr(96)) or not fields[4].startswith(chr(96)):
            continue
        _, card_id, _, parameters, base, upgrade = fields
        defaults = {}
        for name, first, plus in re.findall(r"([DBM])=(-?\d+)(?:→(-?\d+))?", parameters):
            defaults[name] = (int(first), int(plus or first))
        base = base.strip(chr(96))
        catalog[card_id.strip(chr(96))] = (base, upgrade.strip(chr(96)) if upgrade != "—" else base, defaults)
    return catalog


# Reference facts are separate from values that can change in a fight.
REFERENCE = {
    ("potion", "Fire Potion"): "Damage(Target,PotionPotency)",
    ("potion", "Block Potion"): "Block(PotionPotency)",
    ("potion", "Strength Potion"): "Apply(Player,Strength,PotionPotency)",
    ("potion", "Potion Slot"): "EmptySlot",
    ("relic", "SacredBark"): "Scale(PotionPotency,2)",
    ("relic", "Burning Blood"): "On(CombatEnd,Heal(6))",
    ("power", "Strength"): "Modify(OwnerAttackDamage,Add(Stacks))",
    ("power", "Weak"): "Modify(OwnerAttackDamage,Scale(0.75))",
    ("power", "Frail"): "Modify(BlockFromOwnerCards,Scale(0.75))",
    ("power", "Vulnerable"): "Modify(ReceivedAttackDamage,Scale(1.5))",
    ("power", "Curl Up"): "OnceOnUnblockedAttackDamage(GainBlock(Stacks))",
    ("power", "Ritual"): "AtEndOfTurnAfterApplicationTurn(GainStrength(Stacks))",
}
BASE_POTENCY = {"Fire Potion": 20, "Block Potion": 12, "Strength Potion": 2}


def reference_meaning(kind, obj):
    identifier = obj.get("id", "")
    if kind == "card" and identifier in card_catalog():
        base, upgraded, _ = card_catalog()[identifier]
        return Meaning(upgraded if obj.get("upgrades", 0) else base, "catalog")
    program = REFERENCE.get((kind, identifier))
    return Meaning(program, "catalog") if program is not None else None


def tokenize_description(text):
    tokens = re.findall(r"![dbm]!|[-+]?\d+|[a-z]+|[^\w\s]", text.lower())
    values, words = [], []
    for token in tokens:
        if re.fullmatch(r"[-+]?\d+|![dbm]!", token):
            values.append(token.strip("!").upper())
            words.append("#")
        else:
            words.append(token)
    return words, values


def features(kind, words):
    return [kind + "/" + w for w in words] + words + [
        a + " " + b for a, b in zip(["<"] + words, words + [">"])
    ]


class RuleReader(nn.Module):
    """A supervised clause classifier with copied scalar arguments."""
    def __init__(self, vocabulary, templates):
        super().__init__()
        self.templates = templates
        self.index = {feature: i for i, feature in enumerate(vocabulary)}
        self.linear = nn.Linear(len(vocabulary), len(templates))

    def schema(self):
        return {"vocabulary": list(self.index), "templates": self.templates}

    def vector(self, kind, words):
        x = self.linear.weight.new_zeros(len(self.index))
        for f in features(kind, words):
            if f in self.index:
                x[self.index[f]] += 1
        return x

    @torch.no_grad()
    def read_text(self, kind, description):
        programs = []
        for clause in clauses(description):
            if not clause:
                continue
            words, values = tokenize_description(clause)
            if not words or any(w not in self.index for w in words):
                return Meaning(description, "unresolved")
            logits = self.linear(self.vector(kind, words))
            allowed = [i for i, rule in enumerate(self.templates)
                       if kind in rule["kinds"] and len(set(re.findall(r"\$\d+", rule["program"]))) == len(values)]
            if not allowed:
                return Meaning(description, "unresolved")
            # Absolute confidence also includes incompatible labels. A single
            # eligible label must not automatically receive probability one.
            probs = logits.softmax(0)
            best = max(allowed, key=lambda i: probs[i].item())
            if probs[best].item() < 0.85:
                return Meaning(description, "unresolved")
            program = re.sub(r"\$(\d+)", lambda m: values[int(m[1])], self.templates[best]["program"])
            programs.append(program)
        return Meaning(";".join(programs), "predicted") if programs else Meaning("", "unresolved")

    def interpret(self, kind, obj):
        reference = reference_meaning(kind, obj)
        if reference is not None:
            return reference
        description = obj.get("description") or ""
        result = self.read_text(kind, description)
        if kind == "card" and "Attack(" in result.program and obj.get("type") != "ATTACK":
            return Meaning(description, "unresolved")
        return result


def clauses(description):
    plain = re.sub(r"#[brgywp]", "", description, flags=re.I).replace(",", " ")
    return [part for part in re.split(r"\s+NL\s+|[.;]\s*", plain.strip(), flags=re.I) if part]
