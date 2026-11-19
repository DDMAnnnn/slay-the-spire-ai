"""A shared, trainable encoder for one card instance and its ordered public rules."""
import torch
from torch import nn
from torch.nn import functional as F

from card_rules import CARD_TYPES, FIELDS, RULES


class CardEncoder(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.width = width
        self.effect = nn.Linear(sum(map(len, FIELDS.values())) + 2, width)
        self.definition = nn.GRU(width, width, batch_first=True)
        self.sequence = nn.GRU(width * 2, width, batch_first=True)
        self.output = nn.Linear(width + len(CARD_TYPES) + 3, width)

    def effect_vector(self, effect):
        features = []
        for field, vocab in FIELDS.items():
            value = getattr(effect, field)
            if value not in vocab:
                raise ValueError(f"Unsupported {field}: {value}")
            features.extend(float(item == value) for item in vocab)
        features.extend([effect.amount, effect.hits])
        return F.gelu(self.effect(self.effect.weight.new_tensor(features)))

    def forward(self, card, reading):
        rows = []
        zero = self.effect.weight.new_zeros(self.width)
        for effect in reading.effects:
            rules = RULES.get(effect.subject, ())
            meaning = zero
            if rules:
                vectors = torch.stack([self.effect_vector(rule) for rule in rules])
                _, hidden = self.definition(vectors.unsqueeze(0))
                meaning = hidden[0, 0]
            rows.append(torch.cat([self.effect_vector(effect), meaning]))
        summary = zero
        if rows:
            _, hidden = self.sequence(torch.stack(rows).unsqueeze(0))
            summary = hidden[0, 0]
        kind = card.get("type", "UNKNOWN")
        if kind not in CARD_TYPES:
            kind = "UNKNOWN"
        meta = [float(kind == t) for t in CARD_TYPES]
        meta.extend([card["cost"], card["upgrades"], float(bool(reading.unknown))])
        return self.output(torch.cat([summary, zero.new_tensor(meta)]))

