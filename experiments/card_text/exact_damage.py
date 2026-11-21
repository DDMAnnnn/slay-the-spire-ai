"""Regress immediate target HP loss from public pre-action inputs."""
import argparse
import itertools
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from battle.__main__ import load, save
from battle.policy import CombatPolicy
from .run import Batch, HOLDOUT, OPTIONS, card, identity
from .strength import damage, decision, training_data


@torch.no_grad()
def measure(model, double=HOLDOUT):
    single = (2 * double[0] + 2, 1)
    # The single hit leads by one at 1 Strength and trails by one at 3.
    templates = []
    slots = []
    for aliases in itertools.product(range(4), repeat=2):
        for slot in (0, 1):
            rules = (double, single) if slot == 0 else (single, double)
            cards = [dict(card(rule, alias), id=identity(("strength-probe", i), alias))
                     for i, (rule, alias) in enumerate(zip(rules, aliases))]
            templates.append((cards, rules))
            slots.append(slot)
    a_slots = torch.tensor(slots)
    indices = torch.arange(len(templates))
    output = []
    chosen = []
    for s in range(5):
        fights = [decision(cards, rules, s) for cards, rules in templates]
        batch = Batch(model, fights)
        _, outcomes = batch.outputs(model, indices)
        torch.testing.assert_close(outcomes[0], model(fights[0].state, OPTIONS)[1], atol=2e-5, rtol=2e-5)
        predictions = outcomes[:, :, 2]  # existing target HP drop output, in HP units
        cards = torch.stack((predictions[indices, a_slots], predictions[indices, 1 - a_slots]), dim=1)
        truth = torch.tensor([damage(double, s), damage(single, s)])
        output.append(dict(strength=s, truth=truth.tolist(), mean=cards.mean(0).tolist(),
                           mae=(cards - truth).abs().mean(0).tolist(),
                           rounded_exact=(cards.round() == truth).sum(0).tolist()))
        winning_slot = a_slots if truth[0] > truth[1] else 1 - a_slots
        chosen.append(predictions.argmax(1) == winning_slot if s != 2 else None)
    return dict(layouts=len(templates), by_strength=output,
                correct_both_0_4=int((chosen[0] & chosen[4]).sum()),
                correct_both_1_3=int((chosen[1] & chosen[3]).sum()))


def fit_damage(model, optimizer, fights, actions, batches, feedback):
    batch = Batch(model, fights)
    targets = torch.tensor([f.damage_by_action for f in fights], dtype=torch.float32)
    for indices in batches:
        _, outcomes = batch.outputs(model, indices)
        predictions = outcomes[:, :, 2]
        labels = targets[indices]
        if feedback == "played":
            columns = actions[indices, None]
            predictions = predictions.gather(1, columns)
            labels = labels.gather(1, columns)
        loss = F.mse_loss(predictions, labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()


def train(output, seed, schedule, feedback):
    torch.manual_seed(seed)
    model = CombatPolicy(width=32, heads=4, card_input="id_text")
    fights = training_data(schedule)
    generator = torch.Generator().manual_seed(seed + 500)
    actions = torch.randint(3, (len(fights),), generator=generator)
    batches = torch.randint(len(fights), (480, 48), generator=generator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    fit_damage(model, optimizer, fights, actions, batches, feedback)
    stem = output / f"{schedule}-{feedback}-{seed}"
    save(model, stem.with_suffix(".pt"))
    report = measure(load(stem.with_suffix(".pt")))
    stem.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf8")
    print(stem.name, report["correct_both_0_4"], report["correct_both_1_3"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, choices=(19, 23, 41))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    for seed in ([args.seed] if args.seed else [19, 23, 41]):
        for schedule in ("random", "paired"):
            for feedback in ("played", "all"):
                train(args.output, seed, schedule, feedback)
