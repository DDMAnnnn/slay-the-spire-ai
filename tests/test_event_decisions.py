import importlib.util
import unittest

import torch

from battle.policy import CombatPolicy


@unittest.skipUnless(importlib.util.find_spec("sts_env"), "requires the pinned native fork")
class EventDecisionTests(unittest.TestCase):
    def test_only_public_identity_and_run_state_reach_scorer(self):
        import sts_env as sts
        from experiments.headbutt.bridge import block, event_decision, read

        env = sts.Env()
        env.reset(17, 20)
        obs, mask = env.observe(), env.mask()
        d = read(env)
        ids = d.state["event_options"]
        self.assertEqual([c.native for c in d.candidates],
                         [sts.ACT_CHOOSE + i for i in ids])
        self.assertEqual(d.state["event_id"], int(obs[9]) - 1)
        self.assertEqual(len(d.state["deck"]), int(obs[22]))
        player = d.state["player"]
        self.assertEqual((player["current_hp"], player["gold"], player["floor"]),
                         (obs[0], obs[2], obs[3]))
        self.assertEqual(len(player["potions"]), int(obs[16]))
        self.assertEqual(len(player["relics"]), int(obs[23]))
        # Even for Neow, changing payloads describing rewards or costs can't leak
        # them to an ID-only policy. Native action numbers are execution metadata.
        modified = obs.copy()
        block(modified, "choices")[:, 1:5] = 913
        self.assertEqual(event_decision(modified, mask), d)
        model = CombatPolicy(width=16, card_input="hardcoded").eval()
        rows, addresses = model.read_state(d.state)
        for i in ids:
            row = rows[addresses["event_option", i]]
            self.assertIn(f"event_{d.state['event_id']}_option_{i}", row.words)
        self.assertIn(("deck", 0), addresses)
        with torch.no_grad():
            logits, _ = model(d)
            d.candidates.reverse()
            permuted, _ = model(d)
        torch.testing.assert_close(logits.flip(0), permuted)
        for c in d.candidates:
            self.assertTrue(env.clone().step(c.native))
