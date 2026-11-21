"""Compare distinct versus shuffled Strength values on matched logged hands."""
import argparse
import json
from pathlib import Path
import random

import torch

from battle.__main__ import load, save
from battle.policy import CombatPolicy
from .exact_damage import fit_damage, measure
from .run import KNOWN, card
from .strength import decision


def examples(stage, method):
    rng = random.Random(21000 + stage)
    hands, strengths = [], []
    for _ in range(2048):
        rules = rng.sample(KNOWN, 2)
        cards = [card(rule, rng.randrange(4)) for rule in rules]
        hands.extend([(cards, rules)] * 2)
        strengths.extend(rng.sample((0, 2, 4), 2))
    if method != "paired":
        # Same marginal Strength counts and identical hand at every index.
        rng.shuffle(strengths)
    if method == "independent":
        # Ordinary random play gets a fresh hand draw at each decision.
        for i in range(1, len(hands), 2):
            rules = rng.sample(KNOWN, 2)
            hands[i] = ([card(rule, rng.randrange(4)) for rule in rules], rules)
    return [decision(cards, rules, s) for (cards, rules), s in zip(hands, strengths)]


def practice(output, seed, method, pooled=False):
    torch.manual_seed(seed)
    model = CombatPolicy(width=32, heads=4, card_input="id_text")
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    generator = torch.Generator().manual_seed(seed + 500)
    for stage in ((1,) if pooled else (1, 2)):
        fights = examples(stage, method)
        if pooled:
            fights.extend(examples(2, method))
        # Observe the same played action on both visits, including in the control.
        actions = torch.randint(3, (len(fights) // 2,), generator=generator).repeat_interleave(2)
        batches = torch.randint(len(fights), (480, 48), generator=generator)
        fit_damage(model, optimizer, fights, actions, batches, "played")
        budget = "pool480" if pooled else str(stage * 480)
        stem = output / f"{method}-{seed}-{budget}"
        save(model, stem.with_suffix(".pt"))
        report = measure(load(stem.with_suffix(".pt")))
        stem.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf8")
        print(stem.name, report["correct_both_0_4"], report["correct_both_1_3"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, choices=(19, 23, 41))
    parser.add_argument("--method", choices=("random", "paired", "independent"))
    parser.add_argument("--pooled", action="store_true",
                        help="Use all 8192 records from the outset with only 480 updates")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    for seed in ([args.seed] if args.seed else [19, 23, 41]):
        for method in ([args.method] if args.method else ["random", "paired", "independent"]):
            practice(args.output, seed, method, args.pooled)
