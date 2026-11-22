"""Replay one two-target fight and verify Headbutt's two public decisions."""
import argparse
from pathlib import Path

import sts_env as sts
import torch

from battle.__main__ import load
from .bridge import read


def step(env, native):
    if not env.step(native):
        raise RuntimeError(f"Fork rejected native action {native}")


def label(state, option):
    kind, *args = option.split()
    if kind == "end":
        return "End turn"
    zone = "discard" if kind == "retrieve" else "hand"
    slot = int(args[0])
    name = state[zone][slot]["id"]
    target = " on " + state["monsters"][int(args[1])]["id"] if len(args) == 2 else ""
    return f"{zone}[{slot}] {name}{target}"


@torch.inference_mode()
def decide(model, decision):
    options = list(decision.actions)
    logits, _ = model(decision.state, options)
    best = options[logits.argmax().item()]
    for option, value in zip(options, logits.tolist()):
        print(f"  {label(decision.state, option):40} {value:+.4f}" + ("  <- chosen" if option == best else ""))
    return best


def check_draw(env, decision, option):
    selected = decision.state["discard"][int(option.split()[1])]
    step(env, decision.actions[option])
    after = read(env)
    if "selection" in after.state or len(after.state["draw"]) != len(decision.state["draw"]) + 1:
        raise AssertionError("Selection did not return to normal play with one more draw-pile card")
    # A controlled end turn exposes the draw in the new hand. This fixture has no retain.
    step(env, after.actions["end"])
    drawn = read(env).state["hand"][0]
    if (drawn["id"], drawn["upgrades"]) != (selected["id"], selected["upgrades"]):
        raise AssertionError(f"Selected {selected}, first card drawn was {drawn}")
    return drawn["id"]


def run(checkpoint):
    model = load(checkpoint)
    env = sts.Env()
    env.reset(8, 20, character="ironclad", start_cards=[sts.CARD_IDS.index("Headbutt")])
    # Reproducible encounter, ending the opening turn, then a Strike on the medium slime.
    for native in (81, 81, 81, 82, 80, 1):
        step(env, native)
    first = read(env)
    enemies = first.state["monsters"]
    for i, m in enumerate(enemies):
        print(f"Enemy {i}: {m['id']}, {m['current_hp']} HP, {m['intent']} " +
              (f"{m['damage']} x {m['hits']}" if "damage" in m else ""))
    print("1. Play and target")
    play = decide(model, first)
    args = play.split()
    if len(args) != 3 or args[0] != "play" or first.state["hand"][int(args[1])]["id"] != "Headbutt":
        raise AssertionError(f"This checkpoint chose {play}, so this replay did not exercise Headbutt")
    slot, target = map(int, args[1:])
    if sum(o.startswith(f"play {slot} ") for o in first.actions) < 2:
        raise AssertionError("The fixture must offer competing enemy targets")
    step(env, first.actions[play])
    second = read(env)
    hp = [m["current_hp"] for m in enemies]
    hp[target] -= 9
    if [m["current_hp"] for m in second.state["monsters"]] != hp or second.state.get("selection") != "headbutt":
        raise AssertionError("Expected a 9-damage single-target hit followed by a Headbutt prompt")
    print(f"  HP after the hit: {hp}. Energy {first.state['energy']} -> {second.state['energy']}")
    print("2. Choose the next draw")
    retrieve = decide(model, second)
    chosen_slot = int(retrieve.split()[1])
    chosen_id = second.state["discard"][chosen_slot]["id"]
    alternate = next((o for o in second.actions
                      if second.state["discard"][int(o.split()[1])]["id"] != chosen_id), None)
    if alternate is None:
        raise AssertionError("The fixture needs two distinct eligible cards")
    comparison = env.clone()
    drawn = check_draw(env, second, retrieve)
    alternate_draw = check_draw(comparison, second, alternate)
    print(f"First draw after the policy choice: {drawn}. After selecting the alternative: {alternate_draw}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("build/combat-policy/combat.pt"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.checkpoint)
