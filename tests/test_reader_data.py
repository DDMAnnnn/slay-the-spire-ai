import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from battle.__main__ import load, save
from battle.policy import CombatPolicy
from battle.reader import RuleReader, clauses, tokenize_description
from battle.reader_training import DATA, training_rows


def phrasing(row):
    return tuple(tuple(tokenize_description(part)[0]) for part in clauses(row["text"]))


class ReaderDataTests(unittest.TestCase):
    def test_evaluation_splits(self):
        train = training_rows()
        train_keys = {(r["kind"], phrasing(r)) for r in train}
        self.assertEqual(len(train_keys), len(train))
        seen = {phrasing(r) for r in train}
        for split in ("validation", "test"):
            rows = [json.loads(s) for s in (DATA / f"{split}.jsonl").read_text().splitlines()]
            for row in rows:
                if row["group"] == "numbers":
                    self.assertIn((row["kind"], phrasing(row)), train_keys)
                    self.assertFalse(any(r["text"].lower().strip(".") == row["text"].lower().strip(".")
                                         for r in train))
                else:
                    self.assertNotIn(phrasing(row), seen, row)
                    seen.add(phrasing(row))

    def test_checkpoint_inference_independent_of_examples(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy.pt"
            model = CombatPolicy()
            save(model, path)
            with patch("battle.reader_training.training_rows", side_effect=AssertionError):
                restored = load(path)
                self.assertEqual(restored.reader.schema(), model.reader.schema())
                torch.testing.assert_close(restored.reader.linear.weight, model.reader.linear.weight)
                legacy = RuleReader(**json.loads((DATA / "legacy_schema.json").read_text()))
                torch.save({"format": 1, "weights": CombatPolicy(reader=legacy).state_dict()}, path)
                self.assertEqual(load(path).reader.index, legacy.index)
