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


# Each label is a parameterized program. $0 and $1 copy arguments from the
# description, so increasing a damage amount never creates a new output class.
LESSONS = (
    ("card", "Attack(Target,$0)", ("deal 5 damage", "attack for 5 damage")),
    ("card", "Repeat(2,Attack(Target,$0))", ("deal 5 damage twice", "attack twice for 5 damage")),
    ("card", "Repeat($1,Attack(Target,$0))", ("deal 5 damage 3 times", "deal 5 damage to an enemy 3 times")),
    ("potion", "Damage(Target,$0)", ("deal 20 damage", "deal 20 damage to an enemy")),
    ("card potion", "Block($0)", ("gain 5 block", "gain 5 points of block")),
    ("card potion", "Draw($0)", ("draw 2 cards", "draw 1 card")),
    ("card potion", "Apply(Player,Strength,$0)", ("gain 2 strength", "gain 2 stacks of strength")),
    ("card potion", "Apply(Target,Vulnerable,$0)", ("apply 2 vulnerable", "apply 2 stacks of vulnerable")),
    ("relic", "On(CombatEnd,Heal($0))", ("at the end of combat heal 6 hp", "heal 6 hp at the end of combat")),
    ("power", "Modify(OwnerAttackDamage,Add($0))", ("increase attack damage by 2", "increases attack damage by 2")),
    ("enemy", "Repeat($1,Attack(Player,$0))", ("attack for 6 damage 2 times", "deal 6 damage 2 times")),
)

# The current bridge exports IDs but no descriptions for inventory items.
# Reference facts are separate from values that can change in a fight.
REFERENCE = {
    ("potion", "Fire Potion"): "Damage(Target,PotionPotency)",
    ("potion", "Block Potion"): "Block(PotionPotency)",
    ("potion", "Strength Potion"): "Apply(Player,Strength,PotionPotency)",
    ("potion", "Potion Slot"): "EmptySlot",
    ("relic", "SacredBark"): "Scale(PotionPotency,2)",
    ("relic", "Burning Blood"): "On(CombatEnd,Heal(6))",
    ("power", "Strength"): "Modify(OwnerAttackDamage,Add(Stacks))",
}
BASE_POTENCY = {"Fire Potion": 20, "Block Potion": 12, "Strength Potion": 2}


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
    def __init__(self):
        super().__init__()
        self.examples = []
        for label, (kinds, _, phrases) in enumerate(LESSONS):
            for kind in kinds.split():
                for phrase in phrases:
                    self.examples.append((kind, tokenize_description(phrase)[0], label))
        vocab = sorted({f for k, w, _ in self.examples for f in features(k, w)})
        self.index = {f: i for i, f in enumerate(vocab)}
        self.linear = nn.Linear(len(vocab), len(LESSONS))

    def vector(self, kind, words):
        x = self.linear.weight.new_zeros(len(self.index))
        for f in features(kind, words):
            if f in self.index:
                x[self.index[f]] += 1
        return x

    def lesson_loss(self):
        inputs = torch.stack([self.vector(k, w) for k, w, _ in self.examples])
        labels = torch.tensor([label for _, _, label in self.examples], device=inputs.device)
        return nn.functional.cross_entropy(self.linear(inputs), labels)

    def fit_lessons(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.05)
        for _ in range(100):
            loss = self.lesson_loss()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
            allowed = [i for i, (kinds, program, _) in enumerate(LESSONS)
                       if kind in kinds.split() and len(set(re.findall(r"\$\d+", program))) == len(values)]
            if not allowed:
                return Meaning(description, "unresolved")
            # Absolute confidence also includes incompatible labels. A single
            # eligible label must not automatically receive probability one.
            probs = logits.softmax(0)
            best = max(allowed, key=lambda i: probs[i].item())
            if probs[best].item() < 0.85:
                return Meaning(description, "unresolved")
            program = re.sub(r"\$(\d+)", lambda m: values[int(m[1])], LESSONS[best][1])
            programs.append(program)
        return Meaning(";".join(programs), "predicted") if programs else Meaning("", "unresolved")

    def interpret(self, kind, obj):
        identifier = obj.get("id", "")
        if kind == "card" and identifier in card_catalog():
            base, upgraded, _ = card_catalog()[identifier]
            return Meaning(upgraded if obj.get("upgrades", 0) else base, "catalog")
        reference = REFERENCE.get((kind, identifier))
        if reference is not None:
            return Meaning(reference, "catalog")
        description = obj.get("description") or ""
        result = self.read_text(kind, description)
        if kind == "card" and "Attack(" in result.program and obj.get("type") != "ATTACK":
            return Meaning(description, "unresolved")
        return result


def clauses(description):
    plain = re.sub(r"#[brgywp]", "", description, flags=re.I).replace(",", " ")
    return [part for part in re.split(r"\s+NL\s+|[.;]\s*", plain.strip(), flags=re.I) if part]
