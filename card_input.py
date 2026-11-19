"""Inspect exactly the card inputs available in a mailbox decision or a replay."""
import json
from dataclasses import asdict
from pathlib import Path

import torch

from card_encoder import CardEncoder
from card_rules import RULES
from card_translator import Translator


class CardInput:
    def __init__(self, encoder=None):
        self.encoder = CardEncoder() if encoder is None else encoder
        self.translator = Translator()

    def hand(self, state):
        """Vectors follow hand-slot order; the policy can index them by legal move."""
        readings = [self.translator.read(card) for card in state["hand"]]
        vectors = [self.encoder(card, r) for card, r in zip(state["hand"], readings)]
        if not vectors:
            return self.encoder.output.weight.new_empty((0, self.encoder.width)), readings
        return torch.stack(vectors), readings


def inspect(path):
    inputs = CardInput()
    with Path(path).open(encoding="utf-8") as stream, torch.no_grad():
        rows = stream if Path(path).suffix.lower() == ".jsonl" else [stream.read()]
        for raw in rows:
            row = json.loads(raw)
            if row.get("event", "decision") != "decision":
                continue
            state = row["state"]
            vectors, readings = inputs.hand(state)
            for slot, reading in enumerate(readings):
                effects = [asdict(effect) for effect in reading.effects]
                rules = {e.subject: [asdict(rule) for rule in RULES[e.subject]]
                         for e in reading.effects if e.subject in RULES}
                print(json.dumps(dict(turn=state["turn"], slot=slot,
                                      name=state["hand"][slot]["name"], effects=effects,
                                      referenced_rules=rules, unknown=reading.unknown,
                                      vector_width=vectors.shape[1]), ensure_ascii=True))
