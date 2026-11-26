"""Repeat the learned draft on fresh starts, then combine it with both frozen public searches."""
import argparse
import json
from pathlib import Path

from .heart_curriculum import read_model, save_model
from .inventory_search import panel_chunks
from .replay import apply_model
from .train import write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("build/stacked-search"))
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    parent = read_model("build/trajectory-teacher/challenger.json.gz")
    human = read_model("build/human-strategy/challenger.json.gz")
    search = read_model("build/paired-inventory-search/combined.json.gz")
    joint = {**search, "historical_menu": human["historical_menu"]}
    save_model(out / "joint.json.gz", joint)
    dest = out / "report.json"
    result = json.loads(dest.read_text()) if dest.exists() else {"independent_start": 216000000, "matched_start": 194000000}
    for start, name, model in ((216000000, "parent", parent), (216000000, "human", human),
                               (194000000, "human", human), (194000000, "joint", joint)):
        section = result.setdefault(str(start), {})
        if name in section:
            continue
        w = apply_model("first", model)
        section[name] = panel_chunks(out, name, w, start, 32768, args.threads)
        write(dest, result)
        print(start, name, section[name], flush=True)


if __name__ == "__main__":
    main()
