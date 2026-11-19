import unittest

import torch

from card_input import CardInput
from card_rules import Effect, Reading, RULES
from card_translator import EXAMPLES


class CardInputChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = CardInput()
        cls.reader = cls.inputs.translator

    def test_all_supervised_clauses(self):
        for label, *samples in EXAMPLES:
            for sample in samples:
                with self.subTest(sample=sample):
                    result = self.reader.translate(sample)
                    self.assertFalse(result.unknown)
                    expected = label if isinstance(label, tuple) else (label,)
                    self.assertEqual([(e.operation, e.trigger, e.target) for e in result.effects],
                                     [(e.operation, e.trigger, e.target) for e in expected])

    def test_shared_target_and_separate_arguments(self):
        r = self.reader.translate("对所有敌人造成12点伤害，给予2层易伤。")
        self.assertEqual(r.effects, (Effect("damage", "all_enemies", 12, source="attack"),
                                     Effect("apply", "all_enemies", 2, subject="vulnerable")))

    def test_held_out_composition(self):
        expected = Reading((Effect("block", amount=8), Effect("draw", amount=1)))
        for text in ("Gain !B! Block. NL Draw 1 card.", "获得 !B! 点 格挡 。 NL 抽1张牌。"):
            self.assertEqual(self.reader.read(dict(description=text, block=8)), expected)

    def test_unknown_condition_wrap(self):
        result = self.reader.translate("If you have a potion, NL gain 8 Block.")
        self.assertEqual(result.effects, (Effect("unknown"),))
        self.assertTrue(result.unknown)

    def test_body_slam_preview(self):
        for text in ("Deal damage equal to your Block. NL (Deals 12 damage.)",
                     "造成你当前 格挡 值的伤害。 NL （造成 12 点伤害。）"):
            result = self.reader.translate(text)
            self.assertEqual(result, Reading((Effect("damage", "enemy", source="attack",
                                                    amount_from="self_block"),)))

    def test_trigger_and_reference(self):
        result = self.reader.translate("Whenever a card is Exhausted, NL gain 3 Block.")
        self.assertEqual(result.effects[0].trigger, "on_card_exhausted")
        zap, = self.reader.translate("Channel 1 Lightning.").effects
        self.assertEqual(zap.operation, "channel")
        self.assertEqual([(r.trigger, r.amount) for r in RULES[zap.subject]],
                         [("end_turn", 3), ("evoke", 8)])

    def test_card_order_gradient_and_legacy(self):
        base = dict(type="ATTACK", cost=1, upgrades=0)
        hand = [dict(base, description=s) for s in
                ("Deal 7 damage. Draw 1 card.", "Draw 1 card. Deal 7 damage.",
                 "Deal 7 damage. Draw 1 card.", "Channel 1 Lightning.")]
        vectors, readings = self.inputs.hand(dict(hand=hand + [base]))
        self.assertEqual(vectors.shape, (5, 64))
        self.assertTrue(torch.equal(vectors[0], vectors[2]))
        self.assertFalse(torch.equal(vectors[0], vectors[1]))
        self.assertTrue(readings[-1].unknown)
        vectors.sum().backward()
        for p in self.inputs.encoder.parameters():
            self.assertGreater(p.grad.abs().sum().item(), 0)
        self.assertEqual(self.inputs.hand(dict(hand=[]))[0].shape, (0, 64))
