"""Measure description coverage on the user's installed base game."""
import argparse
import json
from zipfile import ZipFile

from card_translator import Translator


def audit(jar, show_unknown=False):
    reader = Translator()
    with ZipFile(jar) as archive:
        for lang in ("eng", "zhs"):
            path = f"localization/{lang}/cards.json"
            cards = json.loads(archive.read(path).decode("utf-8-sig"))
            counts = dict(accepted=0, partial=0, unknown=0)
            for card_id, data in cards.items():
                if not data.get("DESCRIPTION"):
                    continue
                # Sentinels check the parser's coverage, not in-fight quantities.
                result = reader.read(dict(description=data["DESCRIPTION"], damage=7, block=5, magic=2))
                if not result.unknown:
                    counts["accepted"] += 1
                elif any(e.operation != "unknown" for e in result.effects):
                    counts["partial"] += 1
                else:
                    counts["unknown"] += 1
                if show_unknown and result.unknown:
                    print(json.dumps([lang, card_id, result.unknown], ensure_ascii=True))
            print(lang, counts, "total", sum(counts.values()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("game_jar")
    parser.add_argument("--show-unknown", action="store_true")
    args = parser.parse_args()
    audit(args.game_jar, args.show_unknown)
