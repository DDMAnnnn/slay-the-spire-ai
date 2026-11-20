"""Evaluate the text path without consulting object IDs. Run with python -m battle.evaluate_reader."""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

import torch

from .__main__ import load
from .reader import RuleReader, card_catalog
from .reader_training import DATA, fit_reader


def evaluate(reader, examples):
    counts = defaultdict(Counter)
    failures = []
    for case in examples:
        result = reader.read_text(case["kind"], case["text"])
        actual = result.program if result.source == "predicted" else None
        expected = case["expected"]
        if actual == expected:
            outcome = "abstained_correctly" if actual is None else "exact"
        elif actual is None:
            outcome = "abstained"
        else:
            outcome = "wrong_prediction"
        counts[case["group"]][outcome] += 1
        if outcome in ("wrong_prediction", "abstained"):
            failures.append(dict(case, actual=actual))
    return dict(counts), failures


def game_cases(jar, templates):
    # The reader copies numeric literals or D/B/M, and joins clauses with ';'.
    # This is an upper bound: the required arguments must also occur in the text.
    patterns = [re.compile("".join(r"(?:[+-]?\d+|[DBM])" if part.startswith("$") else re.escape(part)
                                   for part in re.split(r"(\$\d+)", rule["program"])))
                for rule in templates if "card" in rule["kinds"]]
    with ZipFile(jar) as game:
        cards = json.loads(game.read("localization/eng/cards.json").decode("utf-8-sig"))
    for card_id, (program, _, _) in card_catalog().items():
        in_schema = all(any(p.fullmatch(step.strip()) for p in patterns) for step in program.split(";"))
        group = "vanilla_in_schema" if in_schema else "vanilla_outside_schema"
        yield {"group": group, "kind": "card", "text": cards[card_id]["DESCRIPTION"],
               "expected": program.replace("; ", ";"),
               "id": card_id}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", nargs="?", type=Path)
    parser.add_argument("--original", action="store_true", help="Reproduce the original 30-example reader")
    parser.add_argument("--game-jar", type=Path)
    parser.add_argument("--errors", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.original == bool(args.checkpoint):
        parser.error("Choose a checkpoint or --original.")
    if args.original:
        torch.manual_seed(19)
        reader = RuleReader(**json.loads((DATA / "legacy_schema.json").read_text(encoding="utf8")))
        fit_reader(reader, original_only=True)
    else:
        reader = load(args.checkpoint).reader
    suites = {s: [json.loads(line) for line in (DATA / f"{s}.jsonl").read_text(encoding="utf8").splitlines()]
              for s in ("validation", "test")}
    if args.game_jar:
        suites["game"] = game_cases(args.game_jar, reader.templates)
    for name, cases in suites.items():
        counts, failures = evaluate(reader, cases)
        print(name, json.dumps(counts, sort_keys=True))
        if args.errors:
            for failure in failures:
                print(json.dumps(failure, ensure_ascii=True))


if __name__ == "__main__":
    main()
