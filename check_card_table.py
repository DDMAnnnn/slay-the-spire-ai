"""Check the hand-authored table against the original game's Java registration."""
import argparse
import ast
import json
import re
import subprocess
from pathlib import Path
from zipfile import ZipFile


def registered_ids(jar):
    library = "com.megacrit.cardcrawl.helpers.CardLibrary"

    def inspect(options, classes):
        return subprocess.check_output(
            ["javap", *options, "-classpath", str(jar), *classes],
            encoding="utf-8", errors="replace",
        )

    code = inspect(["-c", "-p"], [library])
    names = sorted(set(re.findall(
        r"\bnew\s+#\d+\s+// class (com/megacrit/cardcrawl/cards/\w+/\w+)", code
    )))
    cards = inspect(["-constants", "-public"], [n.replace("/", ".") for n in names])
    ids = re.findall(r'\bString ID = "([^"]+)";', cards)
    if not ids or len(names) != len(set(ids)):
        raise ValueError("Could not resolve a unique game ID for every registered class")
    return set(ids)


def check_table(jar, path):
    with ZipFile(jar) as game:
        descriptions = json.loads(game.read("localization/eng/cards.json").decode("utf-8-sig"))
    wanted = registered_ids(jar)
    found = set()
    changed = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = [f.strip() for f in line.strip("|").split("|")]
        if len(fields) != 6 or not fields[1].startswith("`"):
            continue
        name, card_id, cost, params, base, upgrade = fields
        card_id = card_id.strip("`")
        if card_id in found or descriptions[card_id]["NAME"] != name:
            raise ValueError(f"Duplicate ID or mismatched English name: {card_id}")
        found.add(card_id)
        if not re.fullmatch(r"(?:\d+|X|—)(?:→(?:\d+|X|—))?", cost):
            raise ValueError(f"Invalid cost for {card_id}: {cost}")
        if upgrade != "—":
            changed += 1
        for program in [base] + ([upgrade] if upgrade != "—" else []):
            if not (program.startswith("`") and program.endswith("`")):
                raise ValueError(f"Missing program for {card_id}")
            for step in program[1:-1].split(";"):
                tree = ast.parse(step.strip(), mode="eval")
                for node in ast.walk(tree):
                    if not isinstance(node, (ast.Expression, ast.Call, ast.Name, ast.Load,
                                             ast.Constant, ast.UnaryOp, ast.USub)):
                        raise ValueError(f"Unexpected syntax for {card_id}: {step}")
                    if isinstance(node, ast.Name) and node.id in {"D", "B", "M"}:
                        if node.id + "=" not in params:
                            raise ValueError(f"Unbound {node.id} for {card_id}")
    if wanted != found:
        raise ValueError(f"Missing {sorted(wanted-found)}; unexpected {sorted(found-wanted)}")
    print(f"{len(found)} registered cards, {changed} structural upgrade overrides; all rows parsed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("game_jar", type=Path)
    args = parser.parse_args()
    check_table(args.game_jar, Path(__file__).with_name("CARD_TABLE.md"))
