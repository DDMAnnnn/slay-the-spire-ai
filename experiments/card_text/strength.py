"""Does paired practice teach the text policy to use Strength on unseen attacks?"""
import argparse
from dataclasses import dataclass
import itertools
import json
from pathlib import Path
import random

import torch
from torch.nn import functional as F

from battle.__main__ import load, save
from battle.policy import CombatPolicy
from .run import Batch, Fight, HOLDOUT, KNOWN, OPTIONS, card, identity


def damage(rule, strength):
    amount, hits = rule
    return (amount + strength) * hits


@dataclass
class DamageFight(Fight):
    # Evaluator targets. Batch reads only state when constructing model inputs.
    damage_by_action: tuple[int, int, int]


def decision(cards, rules, strength):
    totals = tuple(damage(rule, strength) for rule in rules)
    powers = [{"id": "Strength", "amount": strength}] if strength else []
    enemy = {"id": "practice", "current_hp": 100, "max_hp": 100, "block": 0,
             "powers": [], "intent": "ATTACK", "damage": 8, "hits": 1}
    state = {"player": {"current_hp": 5, "max_hp": 80, "block": 0, "powers": powers, "potions": []},
             "turn": 1, "energy": 1, "hand": cards, "draw": [], "discard": [],
             "exhaust": [], "monsters": [enemy]}
    winner = totals.index(max(totals)) if totals[0] != totals[1] else -1
    return DamageFight(state, winner, "damage", (*totals, 0))


def training_data(schedule):
    rng = random.Random(12000)
    fights = []
    while len(fights) < 4096:
        rules = rng.sample(KNOWN, 2)
        strength = rng.choice((0, 2, 4))
        if damage(rules[0], strength) == damage(rules[1], strength):
            continue
        cards = [card(rule, rng.randrange(4)) for rule in rules]
        fights.append(decision(cards, rules, strength))
    if schedule == "paired":
        # Share the first half exactly. Replace the other half with complete pairs.
        fights = fights[:2048]
        reversals = [(a, b) for a, b in itertools.combinations(KNOWN, 2)
                     if (damage(a, 0) - damage(b, 0)) * (damage(a, 4) - damage(b, 4)) < 0]
        for _ in range(1024):
            rules = list(rng.choice(reversals))
            rng.shuffle(rules)
            cards = [card(rule, rng.randrange(4)) for rule in rules]
            fights.extend(decision(cards, rules, s) for s in (0, 4))
    return fights


def probe():
    cases = {
        "familiar_flip": [((4, 2), (10, 1)), ((6, 2), (14, 1)), ((8, 2), (18, 1))],
        "unseen_flip": [(HOLDOUT, (22, 1))],
        "unseen_stacks_flip": [(HOLDOUT, (22, 1))],
        "unseen_stable": [(HOLDOUT, (18, 1)), (HOLDOUT, (12, 2))],
        "strong_single_stable": [((6, 2), (22, 1))],
    }
    for group, matchups in cases.items():
        fights = []
        for matchup in matchups:
            for aliases in itertools.product(range(4), repeat=2):
                for reverse in (False, True):
                    rules = matchup[::-1] if reverse else matchup
                    cards = [card(rule, alias) for rule, alias in zip(rules, aliases)]
                    if group.startswith("unseen"):
                        # Same unknown IDs at the same slots in both orderings.
                        for slot, item in enumerate(cards):
                            item["id"] = identity(("strength-probe", slot), aliases[slot])
                    strengths = (1, 3) if group == "unseen_stacks_flip" else (0, 4)
                    fights.extend(decision(cards, rules, s) for s in strengths)
        yield group, fights


@torch.no_grad()
def assess(model):
    model.eval()
    results = {}
    for group, fights in probe():
        batch = Batch(model, fights)
        indices = torch.arange(len(fights))
        logits = torch.cat([batch.scores(model, chunk) for chunk in indices.split(64)])
        torch.testing.assert_close(logits[0], model(fights[0].state, OPTIONS)[0], atol=2e-5, rtol=2e-5)
        choices = logits.argmax(1).reshape(-1, 2)
        winners = batch.winners.reshape(-1, 2)
        correct = choices == winners
        # Clamp the higher stack count to the lower one. Its input then matches
        # the first member exactly, so we can reuse that member's prediction.
        clamped = choices[:, :1] == winners
        results[group] = dict(pairs=len(winners), correct_low=int(correct[:, 0].sum()),
                              correct_high=int(correct[:, 1].sum()),
                              both_correct=int(correct.all(1).sum()), clamped_correct=int(clamped.sum()))
    return results


def train(output, seed, schedule):
    torch.manual_seed(seed)
    model = CombatPolicy(width=32, heads=4, card_input="id_text")
    batch = Batch(model, training_data(schedule))
    generator = torch.Generator().manual_seed(seed + 500)
    actions = torch.randint(3, (len(batch.states),), generator=generator)
    batches = torch.randint(len(batch.states), (480, 48), generator=generator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    for indices in batches:
        chosen = actions[indices]
        labels = (chosen == batch.winners[indices]).float()
        logits = batch.scores(model, indices).gather(1, chosen[:, None]).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
    stem = output / f"{schedule}-{seed}"
    save(model, stem.with_suffix(".pt"))
    results = assess(load(stem.with_suffix(".pt")))
    stem.with_suffix(".json").write_text(json.dumps(results, indent=2), encoding="utf8")
    print(schedule, seed, results, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, choices=(19, 23, 41))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    for seed in ([args.seed] if args.seed else [19, 23, 41]):
        for schedule in ("random", "paired"):
            train(args.output, seed, schedule)
