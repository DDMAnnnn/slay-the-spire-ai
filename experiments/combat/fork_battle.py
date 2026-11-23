"""Play the first entire A20 Ironclad battle on the fork, optionally tuning the scorer."""
import argparse
from pathlib import Path

import sts_env as sts
import torch

from battle.__main__ import load, save
from experiments.headbutt.bridge import read
from experiments.headbutt.run import label, step


def fight(model, seed, sample=False, record=False):
    env = sts.Env()
    env.reset(seed, 20, character="ironclad",
              start_cards=[sts.CARD_IDS.index(c) for c in ("Headbutt", "Discovery")])
    # The run model is outside this experiment. Take the first legal out-of-combat option.
    for _ in range(100):
        if env.observe()[7]:
            break
        step(env, int(env.mask().nonzero()[0][0]))
    else:
        raise RuntimeError("No first combat")
    trajectory, frames = [], []
    first_hp = None
    for ordinal in range(400):
        decision = read(env)
        with torch.no_grad():
            logits, _ = model(decision)
            idx = (torch.multinomial(logits.softmax(0), 1).item() if sample else logits.argmax().item())
        chosen = decision.candidates[idx]
        s = decision.state
        if first_hp is None:
            first_hp = s["player"]["current_hp"]
        if sample:
            trajectory.append((decision, idx))
        if record:
            frames.append(dict(state=s, prompt=decision.prompt, chosen=idx,
                               options=[label(s, c) for c in decision.candidates], scores=logits.tolist()))
        step(env, chosen.native)
        if env.done or not env.observe()[7]:
            log = env.episode()
            floor = next(f for f in reversed(log["path"]) if f["fought"])
            return dict(victory=not env.lost, hp=floor["hp_after"], max_hp=floor["max_hp"],
                        start_hp=first_hp, encounter=floor["encounter"], seed=seed,
                        decisions=ordinal + 1), trajectory, frames
    raise RuntimeError(f"Seed {seed} exceeded the combat decision limit")


def tune(model, output):
    from collections import Counter
    counts = Counter()
    for param in model.parameters():
        param.requires_grad_(False)
    params = []
    for module in (model.kind, model.action, model.score):
        for param in module.parameters():
            param.requires_grad_(True)
            params.append(param)
    optimizer = torch.optim.AdamW(params, lr=0.0003)
    for batch in range(4):
        games = []
        for i in range(8):
            end, steps, _ = fight(model, 7000 + batch * 8 + i, sample=True)
            games.append((float(end["victory"]) + end["hp"] / end["max_hp"], steps))
        optimizer.zero_grad()
        total = sum(r for r, _ in games)
        for reward, steps in games:
            advantage = reward - (total - reward) / (len(games) - 1)
            for decision, idx in steps:
                counts[decision.candidates[idx].kind] += 1
                if len(decision.candidates) > 1:
                    logits, _ = model(decision)
                    (-advantage * logits.log_softmax(0)[idx] / len(games)).backward()
        torch.nn.utils.clip_grad_norm_(params, 1)
        optimizer.step()
    save(model, output)
    print("32 training fights, sampled action counts:", dict(counts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("build/combat-policy/combat.pt"))
    parser.add_argument("--tune-to", type=Path)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--replay", type=Path, default=Path("build/combat-policy/battle.html"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(19)
    model = load(args.checkpoint)
    if args.tune_to:
        tune(model, args.tune_to)
        model = load(args.tune_to)
    result, _, frames = fight(model, args.seed, record=True)
    from .viewer import write_replay
    write_replay(args.replay, result, frames)
    print(result)
    print(args.replay.resolve())


if __name__ == "__main__":
    main()
