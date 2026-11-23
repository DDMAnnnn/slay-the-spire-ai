"""Public candidate order, checkpoint migration and the two stage Discovery play."""
import importlib.util
import unittest

import torch

from battle.policy import Candidate, CombatDecision, CombatPolicy
from test_battle import fight


class CandidateTests(unittest.TestCase):
    def test_candidates_keep_order_and_native_id_is_only_bookkeeping(self):
        torch.manual_seed(13)
        model = CombatPolicy(width=16, card_input="hardcoded").eval()
        state = fight()
        left = Candidate(0, "play", ("hand", 0), ("enemy", 0))
        right = Candidate(1, "play", ("hand", 0), ("enemy", 1))
        with torch.no_grad():
            old, _ = model(state, ["play 0 0", "play 0 1"])
            reordered, _ = model(CombatDecision(state, "Play a combat action", [
                Candidate(909, right.kind, right.source, right.target), left]))
        torch.testing.assert_close(reordered, old.flip(0))


@unittest.skipUnless(importlib.util.find_spec("sts_env"), "requires the pinned native fork")
class DiscoveryTests(unittest.TestCase):
    def test_each_offered_card_becomes_the_zero_cost_card_in_hand(self):
        import sts_env as sts
        from experiments.headbutt.bridge import read

        # Enter a real fight, locating an opening hand with Discovery and room to draw.
        for seed in range(20):
            env = sts.Env()
            env.reset(seed, 20, start_cards=[sts.CARD_IDS.index("Discovery")])
            while not env.observe()[7]:
                self.assertTrue(env.step(int(env.mask().nonzero()[0][0])))
            decision = read(env)
            card = next((c for c in decision.candidates if c.kind == "play" and
                         decision.state["hand"][c.source[1]]["id"] == "Discovery"), None)
            if card:
                break
        self.assertIsNotNone(card)
        self.assertTrue(env.step(card.native))
        grid = read(env)
        self.assertIn("zero cost", grid.prompt)
        self.assertEqual([c.native for c in grid.candidates], env.mask().nonzero()[0].tolist())
        self.assertEqual(len(grid.candidates), 3)
        for c in grid.candidates:
            branch = env.clone()
            self.assertEqual(c.source[0], "offer")
            offered = grid.state["offer"][c.source[1]]["id"]
            self.assertTrue(branch.step(c.native))
            after = read(branch)
            self.assertEqual(after.state["hand"][-1]["id"], offered)
            self.assertEqual(after.state["hand"][-1]["cost"], 0)
