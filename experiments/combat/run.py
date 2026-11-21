"""Train a hardcoded-input combat policy from completed A20 starter fights."""
import argparse
import json
from pathlib import Path
import random
import subprocess

import torch
from torch.nn import functional as F

from battle.__main__ import load, save
from battle.policy import CombatPolicy


def episodes(collector, start, count, choose, start_hp=None):
    command = [str(collector.resolve()), str(start), str(count), "0", "--policy"]
    if start_hp is not None:
        command.append(str(start_hp))
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True) as proc:
        try:
            steps = []
            for line in proc.stdout:
                frame = json.loads(line)
                if "victory" in frame:
                    yield steps, frame
                    steps = []
                else:
                    state, options = frame["state"], frame["options"]
                    index = choose(state, options)
                    steps.append((state, options, index))
                    proc.stdin.write(f"{index}\n")
                    proc.stdin.flush()
            if proc.wait():
                raise RuntimeError(f"Combat collector exited with code {proc.returncode}")
        finally:
            if proc.poll() is None:
                proc.kill()


def update(model, optimizer, games):
    returns = [float(end["victory"]) + end["hp"] / end["max_hp"] for _, end in games]
    optimizer.zero_grad()
    for (steps, _), reward in zip(games, returns):
        baseline = (sum(returns) - reward) / (len(returns) - 1)
        for i, (state, options, index) in enumerate(steps):
            if len(options) == 1:
                continue
            scores, outcomes = model(state, options)
            loss = -(reward - baseline) * scores.log_softmax(0)[index]
            # Same signed target-HP-change channel as the recording trainer.
            parts = options[index].split()
            if len(parts) == 3 and i + 1 < len(steps):
                target = int(parts[2])
                next_state = steps[i + 1][0]
                before, after = state["monsters"][target], next_state["monsters"][target]
                if next_state["turn"] == state["turn"] and before["id"] == after["id"]:
                    change = after["current_hp"] - before["current_hp"]
                    loss = loss + 0.01 * F.smooth_l1_loss(outcomes[index, 1], scores.new_tensor(change))
            (loss / len(games)).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()


@torch.no_grad()
def evaluate(collector, start, count, choose, start_hp=None):
    fights = []
    for steps, end in episodes(collector, start, count, choose, start_hp):
        if start_hp is not None:
            assert steps[0][0]["player"]["current_hp"] == start_hp
        end["first_enemy"] = steps[0][0]["monsters"][0]["id"]
        plays = [s["hand"][int(opts[i].split()[1])]["id"] for s, opts, i in steps if opts[i] != "end"]
        end.update(strikes=plays.count("Strike_R"), defends=plays.count("Defend_R"), decisions=len(steps))
        fights.append(end)
    assert len(fights) == count
    return fights


def train(collector, checkpoint):
    model = CombatPolicy(width=32, heads=4, card_input="hardcoded")
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0008)

    @torch.no_grad()
    def sample(state, options):
        if len(options) == 1:
            return 0
        scores, _ = model(state, options)
        return int(torch.multinomial(scores.softmax(0), 1))

    games = []
    train_results = []
    for steps, end in episodes(collector, 22000, 256, sample):
        games.append((steps, end))
        train_results.append(end)
        if len(games) == 8:
            update(model, optimizer, games)
            games.clear()
    assert len(train_results) == 256
    save(model, checkpoint)
    return train_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collector", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--evaluate-hp", type=int, help="Evaluate the existing checkpoint at this starting HP")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(19)
    checkpoint = args.output / "combat.pt"
    report = {}
    if args.evaluate_hp is None:
        report["training"] = train(args.collector, checkpoint)
    policy = load(checkpoint)
    rng = random.Random(19)

    def random_play(state, options):
        plays = [i for i, action in enumerate(options) if action != "end"]
        return rng.choice(plays) if plays else options.index("end")

    def attack_first(state, options):
        # All targeted plays in the starter benchmark are attacks.
        return next((i for i, action in enumerate(options) if len(action.split()) == 3), 0)

    for name, choose in (("random_play", random_play), ("attack_first", attack_first),
                         ("learned", lambda s, opts: opts.index(policy.choose(s, opts)))):
        fights = evaluate(args.collector, 44000, 64, choose, args.evaluate_hp)
        report[name] = fights
        print(name, sum(f["victory"] for f in fights), "wins;",
              sum(f["hp"] for f in fights) / len(fights), "mean remaining HP", flush=True)
    filename = "results.json" if args.evaluate_hp is None else f"results-hp{args.evaluate_hp}.json"
    (args.output / filename).write_text(json.dumps(report, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
