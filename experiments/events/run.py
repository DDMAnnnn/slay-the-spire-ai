"""Learn opaque event-button preferences from matched Act 1 continuations."""
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import random

import sts_env as sts
import torch

from battle.__main__ import load, save
from experiments.headbutt.bridge import block, event_decision, read
from experiments.headbutt.run import step


@dataclass
class Sample:
    seed: int
    decision: object
    outcomes: list


def fixed_action(obs, legal):
    """Public, deterministic continuation shared by every arm, including combat."""
    g = block(obs, "global")[0]
    buttons = [a for a in legal if a >= sts.ACT_CHOOSE]
    choices = block(obs, "choices")
    if g[7]:
        if buttons:
            return sts.ACT_CONFIRM if sts.ACT_CONFIRM in buttons else buttons[0]
        hand = block(obs, "hand")
        foes = block(obs, "monsters")
        plays = [a for a in legal if a < sts.ACT_POTION]
        attacks = [a for a in plays if hand[a // 5, 4]]
        if attacks:
            return min(attacks, key=lambda a: (foes[a % 5, 1] + foes[a % 5, 3], a))
        return plays[0] if plays else sts.ACT_END_TURN
    screen = sts.SCREEN_NAMES[int(g[6])]
    if screen == "MAP":
        # Prefer visible question marks to collect ordinary events. No unrevealed rooms.
        return next((a for a in buttons if sts.ROOM_NAMES[int(choices[a - sts.ACT_CHOOSE, 2])-1]
                     == "EVENT"), buttons[0])
    if screen == "EVENT":
        return buttons[-1] if sts.EVENT_NAMES[int(g[9])-1] != "NEOW" else buttons[0]
    if screen == "SHOP":
        return buttons[-1]
    return buttons[0]


def ordinary(obs):
    g = block(obs, "global")[0]
    if g[7] or sts.SCREEN_NAMES[int(g[6])] != "EVENT":
        return False
    kinds = [sts.CHOICE_KINDS[int(r[0])] for r in block(obs, "choices") if r[5]]
    return len(kinds) >= 2 and all(k == "EVENT_OPTION" for k in kinds)


def play_until(env, stop_at_event=False):
    for _ in range(1500):
        obs = env.observe()
        g = block(obs, "global")[0]
        if env.done or g[4] > 1:
            clear = bool(g[4] > 1 or env.won)
            floor = min(int(g[3]), 17)
            return {"clear": clear, "floor": floor, "return": float(clear) + floor / 17}
        mask = env.mask()
        if stop_at_event and ordinary(obs):
            return event_decision(obs, mask)
        step(env, fixed_action(obs, list(map(int, mask.nonzero()[0]))))
    raise RuntimeError(f"No decision or termination within 1500 steps on seed {env.seed}")


def collect(start, count):
    samples = []
    for seed in range(start, start + count):
        env = sts.Env()
        env.reset(seed, 20, character="ironclad")
        decision = play_until(env, stop_at_event=True)
        if isinstance(decision, dict):
            continue
        outcomes = []
        for c in decision.candidates:
            branch = env.clone()
            step(branch, c.native)
            outcomes.append(play_until(branch))
        samples.append(Sample(seed, decision, outcomes))
    if not samples:
        raise RuntimeError("No ordinary events collected")
    return samples


def fit(model, samples):
    # Fit the existing encoder and scorer from a copy of the combat checkpoint.
    opt = torch.optim.AdamW(model.parameters(), lr=.0006)
    rng = random.Random(41)
    order = list(range(len(samples)))
    for _ in range(6):
        rng.shuffle(order)
        for start in range(0, len(order), 8):
            losses = []
            for idx in order[start:start+8]:
                s = samples[idx]
                returns = torch.tensor([o["return"] for o in s.outcomes])
                if returns.max() == returns.min():
                    continue
                logits, _ = model(s.decision)
                # Each option has its own continuation; equal outcomes convey no ordering.
                margin = returns[:, None] - returns[None, :]
                better = margin > 0
                gap = logits[:, None] - logits[None, :]
                losses.append((margin[better] * torch.nn.functional.softplus(-gap[better])).mean())
            if losses:
                opt.zero_grad()
                torch.stack(losses).mean().backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                opt.step()


def button_means(samples):
    values = defaultdict(list)
    for s in samples:
        for c, outcome in zip(s.decision.candidates, s.outcomes):
            values[s.decision.state["event_id"], c.source[1]].append(outcome["return"])
    return {key: sum(v)/len(v) for key, v in values.items()}


@torch.no_grad()
def evaluate(model, samples, means, initial):
    rows = []
    for s in samples:
        event = s.decision.state["event_id"]
        candidates = s.decision.candidates
        returns = [o["return"] for o in s.outcomes]
        scores = model(s.decision)[0].tolist()
        # Compare relative value within the same encountered event.
        learned = max(range(len(scores)), key=scores.__getitem__)
        avg = [means.get((event, c.source[1]), 0.) for c in candidates]
        lookup = max(range(len(avg)), key=avg.__getitem__)
        frozen = int(initial(s.decision)[0].argmax())
        rows.append(dict(seed=s.seed, event=sts.EVENT_NAMES[event], event_id=event,
                         slots=[c.source[1] for c in candidates], scores=scores, outcomes=s.outcomes,
                         chosen={"learned":learned, "lookup":lookup, "first":0, "frozen":frozen},
                         informative=min(returns) != max(returns)))
    for label in ("learned","lookup","first","frozen","uniform"):
        cleared = floors = best = useful = 0
        for row in rows:
            outcomes = row["outcomes"]
            indices = range(len(outcomes)) if label == "uniform" else [row["chosen"][label]]
            for i in indices:
                w = 1/len(indices)
                cleared += w * outcomes[i]["clear"]
                floors += w * outcomes[i]["floor"]
                if row["informative"]:
                    best += w * (outcomes[i]["return"] == max(o["return"] for o in outcomes))
            useful += row["informative"]
        print(f"{label:8} clears {cleared:.2f}/{len(rows)}, mean floor {floors/len(rows):.2f}, "
              f"best on distinguishable starts {best:.2f}/{useful}")
    return rows


@torch.inference_mode()
def rollout(model, means, seed, use_lookup):
    env = sts.Env()
    env.reset(seed, 20, character="ironclad")
    decisions = []
    pending_grids = set()
    for _ in range(1500):
        obs = env.observe()
        if env.done or obs[4] > 1:
            return {"seed": seed, "clear": bool(obs[4] > 1 or env.won),
                    "floor": min(int(obs[3]), 17), "events": decisions}
        # A greedy policy can toggle the same cards forever in a multi-select.
        # Count the unfinished run explicitly; do not substitute another move.
        if obs[7] and int(block(obs, "player")[0, 7]) == 2:
            grid = obs.tobytes()
            if grid in pending_grids:
                return {"seed": seed, "clear": False, "floor": int(obs[3]), "events": decisions,
                        "truncated": "repeated combat selection"}
            pending_grids.add(grid)
        else:
            pending_grids.clear()
        if obs[7]:
            d = read(env)
            native = d.candidates[int(model(d)[0].argmax())].native
        elif ordinary(obs):
            d = read(env)
            e = d.state["event_id"]
            if use_lookup:
                c = max(d.candidates, key=lambda c: means.get((e, c.source[1]), 0.))
            else:
                c = d.candidates[int(model(d)[0].argmax())]
            native = c.native
            decisions.append({"event": sts.EVENT_NAMES[e], "floor": int(obs[3]),
                              "hp": int(obs[0]), "gold": int(obs[2]),
                              "options": d.state["event_options"], "selected": c.source[1]})
        else:
            native = fixed_action(obs, list(map(int, env.mask().nonzero()[0])))
        step(env, native)
    return {"seed": seed, "clear": False, "floor": int(env.floor), "events": decisions,
            "truncated": "1500-step limit"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("build/combat-policy/fork.pt"))
    parser.add_argument("--output", type=Path, default=Path("build/event-policy"))
    parser.add_argument("--train-seeds", type=int, default=160)
    parser.add_argument("--test-seeds", type=int, default=80)
    parser.add_argument("--rollout-seeds", type=int, default=16)
    args = parser.parse_args()
    if not 0 < args.train_seeds <= 10000 or not 0 < args.test_seeds <= 10000 or args.rollout_seeds < 0:
        parser.error("Use 1..10000 training and test seeds and a nonnegative rollout count")
    torch.set_num_threads(1)
    torch.manual_seed(41)
    train = collect(30000, args.train_seeds)
    test = collect(40000, args.test_seeds)
    print(f"Collected {len(train)} training and {len(test)} test events", flush=True)
    model = load(args.checkpoint)
    fit(model, train)
    args.output.mkdir(parents=True, exist_ok=True)
    save(model, args.output / "events.pt")
    counts = Counter(sts.EVENT_NAMES[s.decision.state["event_id"]] for s in train)
    model = load(args.output / "events.pt")
    means = button_means(train)
    rows = evaluate(model, test, means, load(args.checkpoint))
    rollouts = {}
    for arm in ("learned", "lookup"):
        trials = [rollout(model, means, seed, arm == "lookup")
                  for seed in range(50000, 50000 + args.rollout_seeds)]
        rollouts[arm] = trials
        if trials:
            print(f"Whole act {arm}: {sum(t['clear'] for t in trials)}/{len(trials)} clears,"
                  f" mean floor {sum(t['floor'] for t in trials)/len(trials):.2f}", flush=True)
    report = {"train_seeds": args.train_seeds, "test_seeds": args.test_seeds,
              "train_counts": dict(counts), "test": rows, "rollouts": rollouts}
    (args.output / "results.json").write_text(json.dumps(report, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
