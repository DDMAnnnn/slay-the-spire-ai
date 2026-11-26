"""Turn historical offers into training examples using only acquisitions preceding the offer."""
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

import heart_native as engine
import numpy as np

from .expert_draft import Menus

REV = "411c17ab53b24c3781f5a02c6a4032667bbeda3f"
URL = f"https://raw.githubusercontent.com/karlo-delacruz-ieds/sts_1_winning_percentage/{REV}/A20runs%20JSON.zip"
DIGEST = "bcbed387cfb783cb4065108924b81330f8df3d4e1042d6f7ecc9ccb12c2e63ee"
TOKENS = 2 + engine.card_count + engine.relic_count
STARTERS = {"Strike_R", "Defend_R", "Bash", "AscendersBane"}
FIELDS = ("character_chosen", "ascension_level", "seed_played", "victory", "is_daily", "chose_seed",
          "is_trial", "is_endless", "damage_taken", "card_choices", "boss_relics", "relics_obtained",
          "event_choices", "items_purchased", "item_purchase_floors", "items_purged",
          "items_purged_floors", "floor_reached", "build_version")


def historical_rows(root):
    archive = root / "A20runs.zip"
    if not archive.exists():
        root.mkdir(parents=True, exist_ok=True)
        tmp = archive.with_suffix(".tmp")
        with urllib.request.urlopen(URL, timeout=60) as src, tmp.open("wb") as dest:
            shutil.copyfileobj(src, dest)
        if hashlib.sha256(tmp.read_bytes()).hexdigest() != DIGEST:
            raise ValueError("historical archive checksum mismatch")
        tmp.replace(archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != DIGEST:
        raise ValueError("historical archive checksum mismatch")
    path = root / "ordinary-input-v1.jsonl.gz"
    if not path.exists():
        tmp = path.with_suffix(".tmp")
        with zipfile.ZipFile(archive) as z, z.open("A20runs.json") as f:
            columns = json.load(f)
        with gzip.open(tmp, "wt", encoding="utf8") as out:
            for i, character in columns["character_chosen"].items():
                if character == "IRONCLAD":
                    row = {k: columns[k][i] for k in FIELDS}
                    out.write(json.dumps(row, separators=(",", ":")) + "\n")
        tmp.replace(path)
    with gzip.open(path, "rt", encoding="utf8") as f:
        yield from map(json.loads, f)


def ordinary(r):
    return r["character_chosen"] == "IRONCLAD" and r["ascension_level"] == 20 and not any(
        r.get(k) for k in ("is_daily", "chose_seed", "is_trial", "is_endless"))


class PrefixReader:
    def __init__(self):
        menus = Menus()
        self.cards = {name: 2+i for name, i in menus.cards.items()}
        self.relics = {name: 2+engine.card_count+i for name, i in menus.relics.items()}
        self.skipped = Counter()

    def option(self, name, boss=False):
        if name == "SKIP":
            return (0, 0)
        if name == "Singing Bowl" and not boss:
            return (1, 0)
        item, _, upgrade = name.partition("+")
        token = (self.relics if boss else self.cards).get(item)
        return None if token is None else (token, int(bool(upgrade)))

    def run(self, r):
        counts = Counter()
        def change(name, amount, relic=False):
            base = name.partition("+")[0]
            table = self.relics if relic else self.cards
            if base not in table or base in STARTERS or base == "Burning Blood":
                return
            token = table[base]
            counts[token] = min(1, max(0, counts[token]+amount)) if relic else max(0, counts[token]+amount)

        card_menus, effects = defaultdict(list), defaultdict(list)
        for offer in r["card_choices"]:
            card_menus[int(offer["floor"])].append(offer)
        for e in r["event_choices"]:
            floor = int(e["floor"])
            for field, sign, relic in (("cards_removed",-1,False), ("cards_transformed",-1,False),
                                       ("cards_obtained",1,False), ("relics_lost",-1,True),
                                       ("relics_obtained",1,True)):
                effects[floor].extend((s, sign, relic) for s in e.get(field, []))
        for relic in r["relics_obtained"]:
            effects[int(relic["floor"])].append((relic["key"], 1, True))
        for name, floor in zip(r["items_purchased"], r["item_purchase_floors"]):
            effects[int(floor)].append((name, 1, name in self.relics))
        for name, floor in zip(r["items_purged"], r["items_purged_floors"]):
            effects[int(floor)].append((name, -1, False))

        def example(offer, floor, boss):
            if not offer.get("picked") or not isinstance(offer.get("not_picked"), list):
                self.skipped["incomplete menu"] += 1
                return
            names = [offer["picked"], *offer["not_picked"], "SKIP"]
            if not boss:
                bowl = counts[self.relics["Singing Bowl"]] > 0
                if bowl:
                    names.append("Singing Bowl")
                elif "Singing Bowl" in names:
                    self.skipped["bowl availability unknown"] += 1
                    return
            options = [self.option(name, boss) for name in names]
            if None in options:
                self.skipped["unknown menu item"] += 1
                return
            # The picked item is listed first in run files. Sorting removes that label cue.
            legal = sorted(set(options))
            if len(legal) < 2:
                return
            context = np.zeros(TOKENS, np.uint8)
            for token, count in counts.items():
                context[token] = min(3, count)
            act = int(floor > 17) + int(floor > 34)
            return context, act, int(boss), legal, legal.index(options[0])

        for floor in range(52):
            for offer in card_menus[floor]:
                if floor:
                    ex = example(offer, floor, False)
                    if ex is not None:
                        yield ex
                change(offer.get("picked", ""), 1)
            if floor in (17, 34):
                act = int(floor == 34)
                if act < len(r["boss_relics"]):
                    offer = r["boss_relics"][act]
                    ex = example(offer, floor, True)
                    if ex is not None:
                        yield ex
                    change(offer.get("picked", ""), 1, True)
            # Run histories omit within-floor order between shops, events and rewards.
            # Defer those acquisitions until every labeled offer on that floor is past.
            for args in effects[floor]:
                change(*args)


def pack(examples):
    context, act, kind, menus, pick = zip(*examples)
    options = np.full((len(menus), max(map(len, menus)), 2), -1, np.int16)
    for i, menu in enumerate(menus):
        options[i, :len(menu)] = menu
    return dict(context=np.stack(context), act=np.array(act), kind=np.array(kind),
                options=options, pick=np.array(pick))


def collect(out):
    reader = PrefixReader()
    groups = defaultdict(list)
    counts, versions, seen = Counter(), Counter(), set()
    def add(r, group):
        seed = str(r["seed_played"])
        if not ordinary(r) or seed in seen:
            return
        seen.add(seed)
        groups[group].extend(reader.run(r))
        counts[group] += 1
        versions[str(r.get("build_version"))] += 1
    for r in historical_rows(Path("build/historical-human")):
        if r["floor_reached"] != 57 or not r["victory"] or not r["damage_taken"] or r["damage_taken"][-1]["enemies"] != "The Heart":
            continue
        bucket = int.from_bytes(hashlib.sha256(str(r["seed_played"]).encode()).digest()[:4], "big") % 10
        add(r, "test" if bucket == 9 else "validation" if bucket == 8 else "train")
    for name in ("200-rotating-sample", "chegs", "panacea-ironclad-sample"):
        for path in sorted((Path("build/expert-runs/runs") / name).rglob("*.run")):
            add(json.loads(path.read_bytes()), "streamers")
    for name, items in groups.items():
        np.savez_compressed(out / f"{name}.npz", **pack(items))
    return dict(archive=URL, sha256=DIGEST, runs=dict(counts), decisions={k: len(v) for k,v in groups.items()},
                skipped=dict(reader.skipped), versions=dict(versions))
