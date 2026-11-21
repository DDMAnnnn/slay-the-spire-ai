import copy
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from battle.__main__ import load, save
from battle.policy import CombatPolicy, lexical_tokens, scalars
from experiments.card_text.run import Batch, HOLDOUT, KNOWN, OPTIONS, card, familiar, identity, novel
from experiments.card_text.strength import decision, probe, training_data
from experiments.card_text.broad_pairs import examples


class CardTextTests(unittest.TestCase):
    def test_broad_pair_controls(self):
        def stacks(fight):
            return sum(p["amount"] for p in fight.state["player"]["powers"])

        for stage in (1, 2):
            paired = examples(stage, "paired")
            control = examples(stage, "random")
            independent = examples(stage, "independent")
            self.assertEqual(len(paired), 4096)
            self.assertEqual([f.state["hand"] for f in paired], [f.state["hand"] for f in control])
            self.assertEqual(Counter(map(stacks, paired)), Counter(map(stacks, control)))
            self.assertEqual(list(map(stacks, independent)), list(map(stacks, control)))
            self.assertNotEqual(independent[1].state["hand"], independent[0].state["hand"])
            descriptions = {tuple(sorted(c["description"] for c in f.state["hand"])) for f in paired}
            self.assertEqual(len(descriptions), 105)
            self.assertFalse(any("Deal 10 damage twice." in hand for hand in descriptions))
            self.assertTrue(any(f.winner == -1 for f in paired))
            for first, second in zip(paired[::2], paired[1::2]):
                self.assertEqual(first.state["hand"], second.state["hand"])
                self.assertNotEqual(stacks(first), stacks(second))

    def test_exact_damage_targets_are_separate_from_inputs(self):
        rules = (HOLDOUT, (22, 1))
        cards = [card(r, 0) for r in rules]
        for strength, targets in enumerate(((20,22,0), (22,23,0), (24,24,0), (26,25,0), (28,26,0))):
            fight = decision(cards, rules, strength)
            self.assertEqual(fight.damage_by_action, targets)
            if strength == 2:
                self.assertEqual(fight.winner, -1)
            altered = decision(cards, rules[::-1], strength)
            self.assertEqual(fight.state, altered.state)
            self.assertEqual(altered.damage_by_action, (targets[1], targets[0], 0))

    def test_strength_probe_only_changes_the_status(self):
        for group, fights in probe():
            for low, high in zip(fights[::2], fights[1::2]):
                hidden = copy.deepcopy(high.state)
                low_count, high_count = (1, 3) if group == "unseen_stacks_flip" else (0, 4)
                self.assertEqual(hidden["player"]["powers"], [{"id": "Strength", "amount": high_count}])
                self.assertEqual(low.state["player"]["powers"], [{"id": "Strength", "amount": 1}] if low_count else [])
                hidden["player"]["powers"] = low.state["player"]["powers"]
                self.assertEqual(low.state, hidden)
                self.assertEqual(low.winner != high.winner, group.endswith("_flip"))

    def test_strength_training_budget_and_holdout(self):
        random_fights, paired_fights = (training_data(s) for s in ("random", "paired"))
        self.assertEqual(len(random_fights), len(paired_fights))
        self.assertEqual(random_fights[:2048], paired_fights[:2048])
        for fight in random_fights + paired_fights:
            self.assertNotIn("Deal 10 damage twice.", [c["description"] for c in fight.state["hand"]])
        for low, high in zip(paired_fights[2048::2], paired_fights[2049::2]):
            self.assertEqual(low.state["hand"], high.state["hand"])
            self.assertNotEqual(low.winner, high.winner)

    def test_input_separation_and_visible_numbers(self):
        state = familiar(7,1)[0].state
        other = copy.deepcopy(state)
        state["hand"][0].update(description="Deal !D! damage.", damage=10, base_damage=10)
        other["hand"][0].update(description="Deal !D! damage twice.", damage=6, base_damage=6)
        with patch("battle.policy.card_catalog",side_effect=AssertionError):
            for mode in ("id","id_text"):
                model=CombatPolicy(card_input=mode)
                a=model.read_state(state)[0][1]
                b=model.read_state(other)[0][1]
                self.assertEqual(scalars(a),scalars(b))
                if mode=="id":
                    self.assertEqual(lexical_tokens(a),lexical_tokens(b))
                else:
                    self.assertIn("10",lexical_tokens(a))
                    self.assertIn("twice",lexical_tokens(b))

    def test_unseen_combination_and_swap_controls(self):
        self.assertNotIn(HOLDOUT,KNOWN)
        held,paired=novel(10,8)
        for f in held:
            self.assertIn(identity(HOLDOUT,0),[c["id"] for c in f.state["hand"]])
        for first,second in zip(paired[::2],paired[1::2]):
            self.assertNotEqual(first.winner,second.winner)
            a,b=first.state["hand"],second.state["hand"]
            self.assertEqual([c["id"] for c in a],[c["id"] for c in b])
            self.assertEqual([c["description"] for c in a],[c["description"] for c in reversed(b)])

    def test_batch_and_checkpoint_use_the_game_scorer(self):
        torch.set_num_threads(1)
        model=CombatPolicy(width=32,card_input="id_text").eval()
        fights=familiar(8,4)
        with TemporaryDirectory() as tmp, torch.no_grad():
            path=Path(tmp)/"trained.pt"
            save(model,path)
            restored=load(path)
            self.assertEqual(restored.card_input,"id_text")
            self.assertIsNone(restored.reader)
            actual=Batch(restored,fights).scores(restored,torch.arange(len(fights)))
            expected=torch.stack([model(f.state,OPTIONS)[0] for f in fights])
            torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-5)
