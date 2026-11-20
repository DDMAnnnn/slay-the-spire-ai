import copy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from battle.__main__ import load, save
from battle.policy import CombatPolicy, lexical_tokens, scalars
from experiments.card_text.run import Batch, HOLDOUT, KNOWN, OPTIONS, familiar, identity, novel


class CardTextTests(unittest.TestCase):
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
