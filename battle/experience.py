"""Join each requested action to later visible changes using its decision ticket."""
import json
from dataclasses import dataclass


@dataclass
class Transition:
    state: dict
    options: list
    action: str
    after_player: dict | None
    after_monsters: list | None
    hp_events: tuple = ()
    window_complete: bool = False

    def target_hp_drops(self):
        parts = self.action.split()
        if not self.window_complete or len(parts) != 3:
            return None
        return sum(e["target"] == "enemy" and e.get("slot") == int(parts[2])
                   and e["after_hp"] < e["before_hp"] for e in self.hp_events)

    def hp_deltas(self):
        player_delta = target_delta = None
        if self.after_player is not None:
            player_delta = self.after_player["current_hp"] - self.state["player"]["current_hp"]
        parts = self.action.split()
        if len(parts) == 3 and self.after_monsters is not None:
            slot = int(parts[2])
            before = self.state["monsters"]
            after = self.after_monsters
            if slot < len(after) and [m["id"] for m in before] == [m["id"] for m in after]:
                target_delta = after[slot]["current_hp"] - before[slot]["current_hp"]
        return player_delta, target_delta


def transitions(rows):
    decision = action = None
    schema = 1
    hp_events = []
    for row in rows:
        event = row["event"]
        if event in ("decision", "finish", "start"):
            if decision is not None and action is not None:
                after = row.get("state") or {}
                player = enemies = None
                if event == "decision":
                    player, enemies = after["player"], after["monsters"]
                elif event == "finish" and row.get("outcome") in ("combat ended", "defeat"):
                    player = after if "current_hp" in after else None
                complete = schema >= 2 and player is not None
                yield Transition(decision["state"], decision["options"], action, player, enemies,
                                 tuple(hp_events), complete)
            if event == "start":
                schema = row.get("schema", 1)
            hp_events = []
            decision = row if event == "decision" else None
            action = None
        elif event == "action_requested" and decision and row["ticket"] == decision["ticket"]:
            if row["action"] not in decision["options"]:
                raise ValueError(f"Illegal recorded action {row['action']}")
            action = row["action"]
        elif event == "hp_observed" and action is not None and row.get("ticket") == decision["ticket"]:
            hp_events.append(row)
    if decision is not None and action is not None:
        yield Transition(decision["state"], decision["options"], action, None, None, tuple(hp_events))


def records(path):
    with path.open(encoding="utf8") as stream:
        for line in stream:
            yield json.loads(line)
