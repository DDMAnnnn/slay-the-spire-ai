"""Run the interpreter, inspect action scores, or learn from recorded decisions."""
import argparse
import os
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from agent import HOME, respond
from .experience import records, transitions
from .policy import CombatPolicy, observation_rows


def save(model, path):
    from tempfile import NamedTemporaryFile
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as f:
        tmp = Path(f.name)
    try:
        torch.save({"format": 1, "weights": model.state_dict()}, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def load(path):
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data["format"] != 1:
        raise ValueError("Unsupported battle checkpoint")
    model = CombatPolicy()
    model.load_state_dict(data["weights"])
    return model.eval()


def learn(model, traces, epochs):
    if epochs < 1:
        raise ValueError("epochs must be positive")
    samples = [step for path in traces for step in transitions(records(path))]
    if not samples:
        raise ValueError("No recorded decision with a matching requested action")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)
    for _ in range(epochs):
        for index in torch.randperm(len(samples)).tolist():
            step = samples[index]
            scores, predictions = model(step.state, step.options)
            chosen = step.options.index(step.action)
            loss = F.cross_entropy(scores[None], torch.tensor([chosen]))
            for j, change in enumerate((*step.hp_deltas(), step.target_hp_drops())):
                if change is not None:
                    loss = loss + F.smooth_l1_loss(predictions[chosen, j], scores.new_tensor(change))
            # Retain the supervised language signal when outcomes are sparse.
            loss = loss + model.reader.lesson_loss()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
    counts = [sum(s.hp_deltas()[j] is not None for s in samples) for j in (0, 1)]
    windows = sum(s.target_hp_drops() is not None for s in samples)
    print(f"{len(samples)} demonstrated actions; HP labels: player={counts[0]}, target={counts[1]}, "
          f"target HP-drop windows={windows}")


@torch.no_grad()
def inspect(model, trace, ordinal):
    for i, row in enumerate(r for r in records(trace) if r["event"] == "decision"):
        if i != ordinal:
            continue
        state, options = row["state"], row["options"]
        objects, _ = observation_rows(state, model.reader)
        scores, _ = model(state, options)
        print(f"{len(objects)} objects x {model.tokens.embedding_dim} -> {len(scores)} legal-action logits")
        for obj in objects:
            if obj.role in ("hand", "enemy", "potion", "relic"):
                print("  ", " | ".join(obj.words))
        for action, score in zip(options, scores.tolist()):
            print(f"  {action:14s} {score:+.4f}")
        return
    raise ValueError(f"No decision number {ordinal} in {trace}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("init", "read", "inspect", "train", "play"):
        cmd = sub.add_parser(name)
        cmd.add_argument("checkpoint", type=Path)
        if name == "read":
            cmd.add_argument("kind", choices=("card", "power", "potion", "enemy", "relic"))
            cmd.add_argument("text")
        elif name == "inspect":
            cmd.add_argument("recording", type=Path)
            cmd.add_argument("--decision", type=int, default=0)
        elif name == "train":
            cmd.add_argument("output", type=Path)
            cmd.add_argument("recordings", type=Path, nargs="+")
            cmd.add_argument("--epochs", type=int, default=1)
        elif name == "play":
            cmd.add_argument("--folder", type=Path, default=HOME)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(19)
    if args.command == "init":
        model = CombatPolicy()
        model.reader.fit_lessons()
        save(model, args.checkpoint)
        print("Saved the supervised reader and a randomly initialized combat policy.")
        return
    model = load(args.checkpoint)
    if args.command == "read":
        print(model.reader.read_text(args.kind, args.text))
    elif args.command == "inspect":
        inspect(model, args.recording, args.decision)
    elif args.command == "train":
        learn(model, args.recordings, args.epochs)
        save(model, args.output)
    elif args.command == "play":
        print("Waiting for F8 in combat. Ctrl+C stops the policy.", flush=True)
        ticket = None
        try:
            while True:
                ticket = respond(args.folder, ticket, model.choose)
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
