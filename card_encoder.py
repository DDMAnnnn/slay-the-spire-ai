"""A shared, trainable encoder for one card instance and its ordered public rules."""
import torch
from torch import nn
from torch.nn import functional as F

from card_rules import CARD_TYPES, FIELDS, RULES, visible_description


class CardEncoder(nn.Module):
    def __init__(self, width=64, *, use_text=True):
        super().__init__()
        self.width = width
        self.effect = nn.Linear(sum(map(len, FIELDS.values())) + 2, width)
        self.definition = nn.GRU(width, width, batch_first=True)
        self.sequence = nn.GRU(width * 2, width, batch_first=True)
        self.output = nn.Linear(width + len(CARD_TYPES) + 3, width)
        self.use_text = use_text
        self.byte_embedding = nn.Embedding(256, 24)
        self.text_sequence = nn.GRU(24, width, batch_first=True)

    def text_vector(self, card):
        parts = [visible_description(card)]
        for term, definition in sorted(card.get("glossary", {}).items()):
            parts.append(f"{term}: {definition}")
        data = "\n".join(parts).strip().encode("utf-8")
        if not data:
            return self.output.weight.new_zeros(self.width)
        ids = torch.tensor(list(data), device=self.byte_embedding.weight.device)
        states, _ = self.text_sequence(self.byte_embedding(ids).unsqueeze(0))
        return states.mean(dim=1)[0]

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
        vector = self.output(torch.cat([summary, zero.new_tensor(meta)]))
        return vector + self.text_vector(card) if self.use_text else vector

