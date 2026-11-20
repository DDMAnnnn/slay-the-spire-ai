import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from battle.memory import Cue, Experience, MemoryBank, MemoryInterpreter, cue_for, load_memory, observe, save_memory
from battle.memory_training import remember_through
from battle.policy import CombatPolicy


def state():
    card={"id":"mod:unknown","description":"A new effect","cost":1}
    return {"turn":1,"energy":2,"hand":[card],"draw":[card],"discard":[],"exhaust":[],
            "player":{"current_hp":40,"max_hp":80,"block":0,"powers":[],"potions":[]},
            "monsters":[{"id":"foe","current_hp":20,"max_hp":20,"block":0,"powers":[]}]}


class MemoryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(19)
        self.model=MemoryInterpreter(["target.current_hp","hand.arrivals"])
        self.before=state()
        self.after=copy.deepcopy(self.before)
        self.after["draw"]=[]
        self.after["energy"]=1
        self.after["monsters"][0]["current_hp"]=14
        self.item=observe(self.before,"play 0 0",self.after,"train",2)

    def test_new_identity_copies_observed_effect_even_with_unchanged_hand_size(self):
        bank=MemoryBank()
        cue=self.item.cue
        before,_=self.model.interpret(cue,bank)
        self.assertEqual(self.item.changes["hand.arrivals"],1)
        self.assertNotIn("draw/mod:unknown@0",before)
        bank.remember(self.item)
        pred,_=self.model.interpret(cue,bank)
        self.assertAlmostEqual(pred["hand.arrivals"],1,places=5)
        self.assertAlmostEqual(pred["target.current_hp"],-6,places=5)
        self.assertAlmostEqual(pred["draw/mod:unknown@0"],-1)
        self.after["turn"]=2
        self.assertIsNone(observe(self.before,"play 0 0",self.after,"train",3))

    def test_reload_and_catalogue_free_scoring(self):
        bank=MemoryBank([self.item,self.item])
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/"memory.pt"
            save_memory(path,self.model,bank)
            restored,b=load_memory(path)
            self.assertEqual(len(b.seen),1)
            q=cue_for("card",self.before["hand"][0],self.before,0)
            torch.testing.assert_close(restored.interpret(q,b)[1],self.model.interpret(q,bank)[1])
            policy=CombatPolicy()
            policy.memory_runtime=restored,b
            self.before["hand"][0]["id"]="Pommel Strike"
            with patch("battle.policy.card_catalog",side_effect=AssertionError):
                scores,_=policy(self.before,["play 0 0","end"])
            self.assertEqual(scores.shape,(2,))
            self.assertTrue(scores.isfinite().all())

    def test_direct_examples_keep_their_weight_when_the_related_pool_grows(self):
        cue = self.item.cue
        other = Cue(("card", "other", 0), cue.text, cue.context)
        related = [Experience(other, {}, "other-fight", i) for i in range(12)]
        direct = [Experience(cue, {"hand.arrivals": 1}, "direct-fight", i) for i in range(4)]
        for size in (1, 12):
            predicted, _ = self.model.interpret(cue, MemoryBank(related[:size] + direct))
            self.assertAlmostEqual(predicted["hand.arrivals"], 0.8, places=5)

    def test_only_learn_through_confirmed_observation(self):
        fight="00000000-0000-4000-8000-000000000001"
        def decision(seq,s):
            return dict(event="decision",ticket=f"{fight}-{seq}",state=s,options=["play 0 0"])
        def action(seq):
            return dict(event="action_requested",ticket=f"{fight}-{seq}",action="play 0 0")
        rows=[decision(1,self.before),action(1),decision(3,self.after),action(3),decision(5,self.after)]
        runtime=self.model,MemoryBank()
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/"recordings").mkdir()
            (root/"recordings"/(fight+".jsonl")).write_text("\n".join(map(json.dumps,rows)))
            for seq,n in ((1,0),(3,1),(3,1),(5,2)):
                remember_through(root,f"{fight}-{seq}",runtime,root/"memory.pt")
                self.assertEqual(len(runtime[1].seen),n)
