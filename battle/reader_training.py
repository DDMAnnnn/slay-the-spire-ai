"""Build and fit the text reader. Inference uses the schema in its checkpoint."""
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from .reader import RuleReader, features, tokenize_description

DATA = Path(__file__).parent / "data" / "reader"


def training_rows():
    return [json.loads(line) for line in (DATA / "train.jsonl").read_text(encoding="utf8").splitlines()]


def new_reader():
    templates = json.loads((DATA / "templates.json").read_text(encoding="utf8"))
    vocabulary = sorted({feature for row in training_rows()
                         for feature in features(row["kind"], tokenize_description(row["text"])[0])})
    return RuleReader(vocabulary, templates)


def training_batch(reader, *, original_only=False):
    indices = {t["id"]: i for i, t in enumerate(reader.templates)}
    inputs, labels = [], []
    for row in training_rows():
        if original_only and row["origin"] != "original":
            continue
        label = indices[row["label"]]
        kind = row["kind"]
        if kind not in reader.templates[label]["kinds"]:
            raise ValueError(f"Incompatible training label: {row}")
        words, _ = tokenize_description(row["text"])
        # A legacy checkpoint can only be anchored with features its original
        # reader knows. Keep every original example, skip incompatible additions.
        if all(f in reader.index for f in features(kind, words)):
            inputs.append(reader.vector(kind, words))
            labels.append(label)
    if not inputs:
        raise ValueError("Training corpus and saved reader vocabulary do not overlap.")
    return torch.stack(inputs), torch.tensor(labels, device=reader.linear.weight.device)


def fit_reader(reader, *, original_only=False):
    inputs, labels = training_batch(reader, original_only=original_only)
    optimizer = torch.optim.Adam(reader.parameters(), lr=0.05)
    for _ in range(100):
        loss = F.cross_entropy(reader.linear(inputs), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
