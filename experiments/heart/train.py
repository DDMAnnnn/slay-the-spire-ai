"""Train a full-run public-state utility policy on ordinary A20 Ironclad starts."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import heart_native as engine
import sts_env as sts

# Prior centres and perturbation scales, in the same order as the native Weight enum.
NAMES = "defense kill focus scaling draw debuff elite rest shop hall draft duplicate heal potion cost keyroute swap endgame_draft energy_relic finisher".split()
BASE = np.array([1.87,18.4,1,1.27,4,8.35,2.15,1.95,1.89,.6,3.2,1.65,28.2,1.,.47,9.3,1.1,.36,12,8])
SCALE = np.array([.55,6,2,.55,2,3,2,.8,1,.7,1.5,.8,7,.6,.3,5,.8,.17,4,4])


def write(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf8")
    tmp.replace(path)


def rollouts(w, seed, n, threads):
    return np.asarray(engine.batch(w.tolist(), seed, n, threads))


def statistics(rows):
    return dict(runs=len(rows), mean_floor=round(float(rows[:,0].mean()), 3),
                best_floor=int(rows[:,0].max()), act2=int((rows[:,1]>=2).sum()),
                act3=int((rows[:,1]>=3).sum()), act4=int((rows[:,1]==4).sum()),
                hearts=int(rows[:,4].sum()), truncated=int(rows[:,5].sum()))


def objective(rows):
    # Reaching later acts makes the sparse Heart target visible to training.
    scores = rows[:,0] / 56 + .15*(rows[:,1]-1) + .07*rows[:,2] + 3*rows[:,4] - .3*rows[:,5]
    return float(scores.mean())


def verify(w, seed, destination):
    actions = engine.trace(w.tolist(), seed)
    env = sts.Env()
    env.reset(seed, 20, final_act=True, character="ironclad")
    keys = [False]*3
    for i, a in enumerate(actions):
        obs = env.observe()
        mask = env.mask()
        action = int(np.argmax(engine.scores(w.tolist(), obs, mask)))
        if action != a or not mask[a] or not env.step(a):
            raise AssertionError(f"Policy replay diverged at decision {i}")
        keys = [k or bool(v) for k, v in zip(keys, obs[12:15])]
    episode = env.episode()
    fights = [f for f in episode["path"] if f["fought"]]
    if not (env.done and env.won and all(keys) and episode["act"] == 4 and
            fights[-1]["encounter"] == "The Heart" and fights[-1]["hp_after"] > 0 and
            actions == episode["actions"] and episode["ascension"] == 20 and
            not episode.get("start_cards") and not episode.get("start_thin")):
        raise AssertionError("Run did not meet the A20 Heart victory conditions")
    write(destination, dict(episode=episode, keys=keys, weights=w.tolist(),
                            event_table=json.loads((destination.parent/"event-table.json").read_text()),
                            policy_source_sha256=sha256(Path(__file__).with_name("policy.cpp").read_bytes()).hexdigest()))
    print(f"Verified Heart victory: seed {seed}, {len(actions)} actions, {fights[-1]['hp_after']} HP", flush=True)


def train(args):
    rng = np.random.default_rng(74281)
    center = BASE.copy()
    best = center.copy()
    validation = rollouts(best, 6200000, args.validation, args.threads)
    train_bank = engine.Bank(center.tolist(), 4500000, 16000, 96, 2)
    val_bank = engine.Bank(center.tolist(), 6400000, 16000, 96, 2)
    def late(w, bank):
        return objective(np.asarray(bank.batch(w.tolist(), args.threads))) if len(bank) else 0.
    valid_score = objective(validation) + late(best, val_bank)
    history = [dict(stage="prior", validation=statistics(validation))]
    print("Prior", history[-1], flush=True)
    for gen in range(args.generations):
        seed = 3000000 + gen*2048
        noise = rng.normal(size=(args.population//2, len(BASE)))
        candidates = np.maximum(.02,center + SCALE*np.vstack([noise,-noise])*(.9**gen))
        scores = [objective(rollouts(w, seed, args.batch, args.threads)) + late(w, train_bank) for w in candidates]
        elites = np.argsort(scores)[-max(2,len(scores)//4):]
        center = .25*center + .75*candidates[elites].mean(0)
        for name, trial in (("centre",center), ("winner",candidates[elites[-1]])):
            vals = rollouts(trial, 6200000, args.validation, args.threads)
            s = objective(vals) + late(trial, val_bank)
            if s > valid_score:
                best, valid_score = trial.copy(), s
        history.append(dict(generation=gen+1, validation_objective=valid_score,
                            train_best=max(scores), train_median=float(np.median(scores))))
        write(args.out/"policy.json", dict(names=NAMES, weights=best.tolist(), history=history))
        print(history[-1], flush=True)
    return best


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("build/heart-policy"))
    p.add_argument("--generations",type=int,default=5)
    p.add_argument("--population",type=int,default=12)
    p.add_argument("--batch",type=int,default=96)
    p.add_argument("--validation",type=int,default=192)
    p.add_argument("--threads",type=int,default=6)
    p.add_argument("--load",type=Path)
    p.add_argument("--eval-start",type=int,default=9000000)
    p.add_argument("--eval-count",type=int,default=2048)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    table_path = (args.load.parent if args.load else args.out) / "event-table.json"
    if table_path.exists():
        table = json.loads(table_path.read_text())
    else:
        import torch
        from experiments.events.run import collect, load
        torch.set_num_threads(1)
        teacher = load(Path("build/event-policy/events.pt"))
        values = np.zeros((len(sts.EVENT_NAMES),16))
        counts = np.zeros_like(values)
        with torch.inference_mode():
            for sample in collect(30000,160):
                d = sample.decision
                logits = teacher(d)[0]
                for candidate, logit in zip(d.candidates, logits-logits.mean()):
                    key = d.state["event_id"], candidate.source[1]
                    values[key] += float(logit)
                    counts[key] += 1
        table = (values / counts.clip(min=1)).tolist()
    write(args.out/"event-table.json", table)
    engine.set_event_table(table)
    w = np.asarray(json.loads(args.load.read_text())["weights"]) if args.load else train(args)
    all_rows = []
    win_seed = None
    for offset in range(0,args.eval_count,128):
        rows = rollouts(w, args.eval_start+offset,min(128,args.eval_count-offset),args.threads)
        all_rows.extend(rows.tolist())
        wins = np.flatnonzero(rows[:,4])
        if win_seed is None and len(wins):
            win_seed = args.eval_start+offset+int(wins[0])
            verify(w, win_seed, args.out/"heart-win.json")
        print(f"Tested {len(all_rows)}: {statistics(np.asarray(all_rows))}", flush=True)
        write(args.out/"evaluation.json",dict(first_seed=args.eval_start, first_win=win_seed,
              fields=["floor","act","keys","decisions","heart","truncated"], rows=all_rows))


if __name__ == "__main__":
    main()
