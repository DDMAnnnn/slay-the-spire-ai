"""Rotate the unseen two-hit attack at the original matched training budget."""
import argparse
import json
from pathlib import Path

import torch

from battle.__main__ import load, save
from battle.policy import CombatPolicy
from .broad_pairs import examples, train_block
from .exact_damage import measure
from .run import KNOWN, HOLDOUT


def training_rules(amount):
    return sorted(set(KNOWN + [HOLDOUT]) - {(amount, 2)})


def run(output, amount, seed, method):
    fights = examples(1, method, training_rules(amount))
    torch.manual_seed(seed)
    model = CombatPolicy(width=32, heads=4, card_input="id_text")
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    generator = torch.Generator().manual_seed(seed + 500)
    train_block(model, optimizer, generator, fights)
    stem = output / f"hold{amount}-{method}-{seed}"
    save(model, stem.with_suffix(".pt"))
    report = measure(load(stem.with_suffix(".pt")), (amount, 2))
    stem.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf8")
    print(stem.name, report["correct_both_1_3"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--amount", type=int, choices=(4, 6, 8, 10))
    parser.add_argument("--seed", type=int, choices=(19, 23, 41))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    for amount in ([args.amount] if args.amount else (4, 6, 8, 10)):
        for seed in ([args.seed] if args.seed else (19, 23, 41)):
            for method in ("paired", "random", "independent"):
                run(args.output, amount, seed, method)
