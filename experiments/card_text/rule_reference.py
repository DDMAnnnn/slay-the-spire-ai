"""Reference and parsed-input control for the single-target Strength benchmark."""
import argparse
import json
from pathlib import Path
import re

import torch

from battle.policy import CombatPolicy
from .broad_pairs import examples, train_block
from .exact_damage import measure, measure_predictions
from .holdout_panel import training_rules


def attack_fields(description):
    match = re.fullmatch(r"Deal (\d+) damage( twice)?\.", description)
    if match is None:
        raise ValueError(f"Unsupported benchmark description: {description!r}")
    return int(match[1]), 2 if match[2] else 1


def reference_damage(state):
    # This fixture has no other modifiers, no Block and enough HP for every hit.
    player_powers = state["player"]["powers"]
    strength = sum(p["amount"] for p in player_powers if p["id"] == "Strength")
    estimates = []
    for card in state["hand"]:
        base, hits = attack_fields(card["description"])
        estimates.append((base + strength) * hits)
    return [*estimates, 0]


class ParsedAttackPolicy(CombatPolicy):
    def __init__(self):
        super().__init__(width=32, heads=4, card_input="id_text")

    def read_state(self, state):
        rows, addresses = super().read_state(state)
        for slot, card in enumerate(state["hand"]):
            row = rows[addresses["hand", slot]]
            row.numbers["D"], row.numbers["hits"] = attack_fields(card["description"])
        return rows, addresses


def evaluate_reference(amount):
    def predict(fights):
        return torch.tensor([reference_damage(f.state) for f in fights], dtype=torch.float32)

    return measure_predictions(predict, (amount, 2))


def train_parsed(output, amount, seed):
    # Reuse the matched shuffled-Strength schedule and the exact logged actions.
    torch.manual_seed(seed)
    model = ParsedAttackPolicy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    generator = torch.Generator().manual_seed(seed + 500)
    train_block(model, optimizer, generator, examples(1, "random", training_rules(amount)))
    path = output / f"parsed-{amount}-{seed}.pt"
    torch.save(model.state_dict(), path)
    restored = ParsedAttackPolicy()
    restored.load_state_dict(torch.load(path, weights_only=True))
    report = measure(restored.eval(), (amount, 2))
    path.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf8")
    print(path.stem, report["correct_both_1_3"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, choices=(19, 23, 41))
    parser.add_argument("--reference-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    for amount in (4, 6, 8, 10):
        ref = evaluate_reference(amount)
        (args.output / f"reference-{amount}.json").write_text(json.dumps(ref, indent=2), encoding="utf8")
        if args.reference_only:
            continue
        for seed in ([args.seed] if args.seed else (19, 23, 41)):
            train_parsed(args.output, amount, seed)
