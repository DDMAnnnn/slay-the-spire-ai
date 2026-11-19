"""Run the wiring policy, or review one Spire Combat AI decision trace."""
import argparse
import json
import os
import time
from pathlib import Path

HOME = Path.home() / "SpireCombatAI"


def choose(state, options):
    """Exercise card play and a simple targeted potion without spending every bottle."""
    for option in options:
        parts = option.split()
        if parts[0] == "potion" and len(parts) == 3:
            if state["player"]["potions"][int(parts[1])] == "Fire Potion":
                return option
    return next((move for move in options if move.startswith("play ")), "end")


def respond(folder, previous=None, policy=choose):
    try:
        decision = json.loads((folder / "decision.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return previous
    ticket = decision["ticket"]
    if ticket == previous:
        return previous
    options = tuple(decision["options"])
    action = policy(decision["state"], options)
    if action not in options:
        raise ValueError(f"Unavailable action: {action}")
    temp = folder / f"action-{os.getpid()}.tmp"
    temp.write_text(json.dumps(dict(ticket=ticket, action=action)), encoding="utf-8")
    temp.replace(folder / "action.json")
    return ticket


def review(path, full):
    last = None
    with path.open(encoding="utf-8") as stream:
        for i, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"Unreadable row {i}; the trace may be truncated.")
                break
            last = row["event"]
            if full or last != "decision":
                print(json.dumps(row, ensure_ascii=True))
                continue
            state = row["state"]
            hand = [card["name"] for card in state["hand"]]
            foes = [(m["name"], m["current_hp"], m.get("intent", "hidden"))
                    for m in state["monsters"]]
            print(f'#{row["seq"]} turn {state["turn"]}: HP {state["player"]["current_hp"]}, '
                  f'block {state["player"]["block"]}, energy {state["energy"]}')
            print("  hand:", hand, "enemies:", foes)
            print("  potions:", state["player"]["potions"], "choices:", row["options"])
    if last != "finish":
        print("Incomplete fight: no observed finish.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    live = sub.add_parser("play")
    live.add_argument("--folder", type=Path, default=HOME)
    replay = sub.add_parser("replay")
    replay.add_argument("file", type=Path)
    replay.add_argument("--full", action="store_true")
    encode = sub.add_parser("encode", help="inspect card translations and encoder input")
    encode.add_argument("file", type=Path)
    args = parser.parse_args()
    if args.mode == "encode":
        from card_input import inspect
        inspect(args.file)
    elif args.mode == "replay":
        review(args.file, args.full)
    else:
        print("Waiting for F8 in combat. Ctrl+C stops the policy.", flush=True)
        previous = None
        try:
            while True:
                previous = respond(args.folder, previous)
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
