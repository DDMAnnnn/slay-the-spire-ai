import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from battle.experience import transitions
from battle.policy import CombatPolicy, observation_rows
from battle.reader import card_catalog
from agent import respond


def fight():
    card = dict(id="Twin Strike", type="ATTACK", cost=1, damage=5, upgrades=0)
    enemy = dict(id="foe", current_hp=12, max_hp=20, block=0, powers=[], intent="BUFF")
    return dict(player=dict(current_hp=37, max_hp=80, block=3, powers=[], potions=["Fire Potion"],
                            relics=[dict(id="Burning Blood", counter=-1)]),
                hand=[card, card.copy()], draw=[card.copy(), dict(id="Defend_R", cost=1)],
                discard=[], exhaust=[], monsters=[enemy, dict(enemy, current_hp=3)],
                energy=2, turn=2)


class BattleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(19)
        cls.model = CombatPolicy().eval()
        cls.model.reader.fit_lessons()

    def test_semantics(self):
        reader = self.model.reader
        self.assertEqual(len(card_catalog()), 370)
        self.assertEqual(reader.interpret("card", {"id": "Twin Strike"}).program,
                         "Repeat(2,Attack(Target,D))")
        self.assertNotEqual(reader.interpret("card", {"id": "Armaments"}).program,
                            reader.interpret("card", {"id": "Armaments", "upgrades": 1}).program)
        self.assertEqual(reader.read_text("potion", "Deal 37 damage.").program, "Damage(Target,37)")
        self.assertEqual(reader.read_text("card", "Deal 37 damage 4 times.").program,
                         "Repeat(4,Attack(Target,37))")
        self.assertEqual(reader.read_text("card", "Gain 8 Flux.").source, "unresolved")
        self.assertEqual(reader.read_text("card", "Draw 2 cards. Gain 5 block.").program, "Draw(2);Block(5)")

    def test_equivariance_and_duplicate_cards(self):
        state = fight()
        options = ["play 0 0", "play 0 1", "play 1 0", "potion 0 0", "end"]
        with torch.no_grad():
            scores, _ = self.model(state, options)
            self.assertEqual(scores.shape, (5,))
            torch.testing.assert_close(scores[0], scores[2])
            state["draw"].reverse()
            torch.testing.assert_close(scores, self.model(state, options)[0])
            state["monsters"].reverse()
            corresponding = ["play 0 1", "play 0 0", "play 1 1", "potion 0 1", "end"]
            torch.testing.assert_close(scores, self.model(state, corresponding)[0])
            torch.testing.assert_close(scores.flip(0), self.model(state, corresponding[::-1])[0])
            self.assertIn(self.model.choose(state, options), options)
            self.assertEqual(self.model(state, ["end"])[0].shape, (1,))

    def test_snapshots_and_feedback(self):
        before = fight()
        after = copy.deepcopy(before)
        after["monsters"][0]["current_hp"] -= 5
        opts = ["play 0 0", "end"]
        first = dict(event="decision", ticket="a", state=before, options=opts)
        choice = dict(event="action_requested", ticket="a", action=opts[0])
        second = dict(first, state=after, ticket="b")
        step, = transitions([first, dict(choice, ticket="stale"), choice, second])
        self.assertEqual(step.hp_deltas(), (0, -5))
        partial, = transitions([first, choice])
        self.assertEqual(partial.hp_deltas(), (None, None))
        self.assertEqual(list(transitions([first, dict(choice, ticket="stale"), second])), [])
        hp = dict(event="hp_observed", ticket="a", target="enemy", slot=0, before_hp=12, after_hp=7)
        step, = transitions([dict(event="start", schema=2), first, choice, hp,
                             dict(hp, before_hp=7, after_hp=2), second])
        self.assertEqual(step.target_hp_drops(), 2)
        self.assertIsNone(partial.target_hp_drops())

    def test_mailbox_uses_exact_option(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            options = ["potion 0 1", "play 1 0", "play 0 1", "end"]
            (folder / "decision.json").write_text(json.dumps(dict(ticket="x", state=fight(), options=options)))
            self.assertEqual(respond(folder, policy=self.model.choose), "x")
            reply = json.loads((folder / "action.json").read_text())
            self.assertIn(reply["action"], options)
            self.assertEqual(reply["ticket"], "x")

    def test_gradients_reach_reader_on_inferred_rules(self):
        state = fight()
        state["hand"][0] = dict(id="mod:Double", type="ATTACK", description="Deal 13 damage twice.")
        state["monsters"][0].pop("intent")
        rows, _ = observation_rows(state, self.model.reader)
        self.assertIn("hidden", next(r for r in rows if r.role == "enemy").words)
        self.model.zero_grad()
        _, effects = self.model(state, ["play 0 0", "end"])
        effects[0, 1].backward()
        for param in (self.model.reader.linear.weight, self.model.program.weight_ih_l0,
                      self.model.battle.layers[0].self_attn.in_proj_weight):
            self.assertGreater(param.grad.abs().sum().item(), 0.)
