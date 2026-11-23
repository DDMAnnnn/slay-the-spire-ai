"""Replay one two-target fight and verify Headbutt's two public decisions."""
import argparse
from pathlib import Path

import sts_env as sts
import torch

from battle.__main__ import load
from .bridge import PLAY, read


def step(env, native):
    if not env.step(native):
        raise RuntimeError(f"Fork rejected native action {native}")


def label(state, option):
    if not option.source:
        return {"end": "End turn", "skip": "Skip", "confirm": "Confirm selection"}[option.kind]
    zone, slot = option.source
    name = state["player"]["potions"][slot] if zone == "potion" else state[zone][slot]["id"]
    target = " on " + state["monsters"][option.target[1]]["id"] if option.target else ""
    verb = {"play":"Play", "select":"Choose", "potion":"Use", "discard_potion":"Discard"}[option.kind]
    return f"{verb} {name} ({zone} {slot + 1}){target}"


@torch.inference_mode()
def decide(model, decision):
    options = decision.candidates
    logits, _ = model(decision)
    best = options[logits.argmax().item()]
    for option, value in zip(options, logits.tolist()):
        print(f"  {label(decision.state, option):40} {value:+.4f}" + ("  <- chosen" if option == best else ""))
    return best


def check_draw(env, decision, option):
    selected = decision.state["discard"][option.source[1]]
    step(env, option.native)
    after = read(env)
    if after.prompt != PLAY or len(after.state["draw"]) != len(decision.state["draw"]) + 1:
        raise AssertionError("Selection did not return to normal play with one more draw-pile card")
    # A controlled end turn exposes the draw in the new hand. This fixture has no retain.
    step(env, next(c.native for c in after.candidates if c.kind == "end"))
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
    if play.kind != "play" or not play.target or first.state["hand"][play.source[1]]["id"] != "Headbutt":
        raise AssertionError(f"This checkpoint chose {play}, so this replay did not exercise Headbutt")
    target = play.target[1]
    if sum(o.source == play.source and o.target is not None for o in first.candidates) < 2:
        raise AssertionError("The fixture must offer competing enemy targets")
    step(env, play.native)
    second = read(env)
    hp = [m["current_hp"] for m in enemies]
    hp[target] -= 9
    if [m["current_hp"] for m in second.state["monsters"]] != hp or second.prompt != "Headbutt choose discard to draw pile top":
        raise AssertionError("Expected a 9-damage single-target hit followed by a Headbutt prompt")
    print(f"  HP after the hit: {hp}. Energy {first.state['energy']} -> {second.state['energy']}")
    print("2. Choose the next draw")
    retrieve = decide(model, second)
    chosen_slot = retrieve.source[1]
    chosen_id = second.state["discard"][chosen_slot]["id"]
    alternate = next((o for o in second.candidates
                      if second.state["discard"][o.source[1]]["id"] != chosen_id), None)
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
