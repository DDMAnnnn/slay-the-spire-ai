"""Train a full-run public-state utility policy on ordinary A20 Ironclad starts."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
import time

import numpy as np
import heart_native as engine
import sts_env as sts

# Prior centres and perturbation scales, in the same order as the native Weight enum.
NAMES = "defense kill focus scaling draw debuff elite rest shop hall draft duplicate heal potion cost keyroute swap endgame_draft energy_relic finisher lookahead".split()
BASE = np.array([2.24,17.87,.03,1.31,6.70,10.12,2.24,3.44,2.38,.06,2.52,1.93,30.9,1.,.57,4,2.42,.396,12.03,3.97,2])
SCALE = np.array([.55,6,2,.55,2,3,2,.8,1,.7,1.5,.8,7,.6,.3,5,.8,.17,4,4,1.5])


def write(path, obj):
    # On Windows a concurrent progress reader can briefly prevent replacement.
    with NamedTemporaryFile(mode="w", encoding="utf8", dir=path.parent, suffix=".tmp", delete=False) as fp:
        tmp = Path(fp.name)
        json.dump(obj, fp, indent=2)
    try:
        for attempt in range(8):
            try:
                tmp.replace(path)
                return
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(.025 * 2**attempt)
    finally:
        tmp.unlink(missing_ok=True)


def configure(source, destination):
    """Restore every fitted component before evaluating a checkpoint directory."""
    from .replay import set_combat_preferences, set_opening, set_draft, set_late_combat, set_final_weights, set_actor, set_forecast, set_boss_tactics, set_recovery, set_edits, set_scenarios, set_route, set_opening_search, set_teacher, set_expert, set_curriculum, set_inventory_value, set_choice_attention
    for filename, setter, default in (
        ("opening-search",set_opening_search,None), ("route",set_route,None), ("scenarios", set_scenarios, None), ("deck-edits", set_edits, None), ("recovery", set_recovery, None), ("boss-tactics", set_boss_tactics, None), ("forecast", set_forecast, None), ("direct-actor", set_actor, None), ("trajectory-teacher", set_teacher, None), ("boss-draft", engine.set_boss_draft, []), ("expert-prior", set_expert, None), ("curriculum-actor", set_curriculum, None), ("inventory-value", set_inventory_value, None), ("choice-attention", set_choice_attention, None),
        ("late-combat-last-act", engine.set_late_combat_last_act, 4), ("final-weights", set_final_weights, None),
        ("late-combat", set_late_combat, None),
        ("draft-context", set_draft, None), ("neow", set_opening, None),
        ("combat-preferences", set_combat_preferences, {}),
        ("turn-depth", engine.set_turn_depth, 2),
        ("first-boss-relics", engine.set_first_boss_bonus, [0.] * engine.relic_count),
    ):
        path = source/(filename + ".json")
        value = json.loads(path.read_text()) if path.exists() else default
        setter(value)
        if source.resolve() != destination.resolve():
            write(destination/path.name, value)
    for filename, setter in (
        ("event-table", engine.set_event_table), ("choice-table", engine.set_choice_table),
        ("combat-head", lambda head: engine.set_head(*head)),
        ("late", lambda late: engine.set_late(late["act"], late["weights"])),
        ("synergies", engine.set_synergies),
    ):
        path = source / (filename + ".json")
        if path.exists():
            data = json.loads(path.read_text())
            setter(data)
            if source.resolve() != destination.resolve():
                write(destination/path.name, data)
    from .replay import set_public_belief, set_inventory_search, set_historical_menu, set_historical_gate, set_resource_policy, set_deck_forecast, set_learned_map, set_route_worlds, set_human_events, set_strategic_apprentice
    path=source/"public-belief.json"
    config=json.loads(path.read_text()) if path.exists() else None
    set_resource_policy(None)
    set_deck_forecast(None)
    set_learned_map(None)
    set_route_worlds(None)
    set_human_events(None)
    set_strategic_apprentice(None)
    set_public_belief(config)
    if source.resolve()!=destination.resolve():
        write(destination/path.name,config)
    path=source/"historical-menu.json"
    menu=json.loads(path.read_text()) if path.exists() else None
    set_historical_menu(menu)
    if source.resolve()!=destination.resolve():
        write(destination/path.name,menu)
    path=source/"route-worlds.json"
    worlds=json.loads(path.read_text()) if path.exists() else None
    set_route_worlds(worlds)
    apprentice_path=source/"strategic-apprentice.json"
    apprentice=json.loads(apprentice_path.read_text()) if apprentice_path.exists() else None
    set_strategic_apprentice(apprentice)
    if source.resolve()!=destination.resolve():
        write(destination/apprentice_path.name,apprentice)
    event_path=source/"human-events.json"
    event_model=json.loads(event_path.read_text()) if event_path.exists() else None
    set_human_events(event_model)
    if source.resolve()!=destination.resolve():
        write(destination/event_path.name,event_model)
    if source.resolve()!=destination.resolve():
        write(destination/path.name,worlds)
    path=source/"map-policy.json"
    map_policy=json.loads(path.read_text()) if path.exists() else None
    set_learned_map(map_policy)
    if source.resolve()!=destination.resolve():
        write(destination/path.name,map_policy)
    path=source/"deck-outcome.json"
    forecast=json.loads(path.read_text()) if path.exists() else None
    set_deck_forecast(forecast)
    if source.resolve()!=destination.resolve():
        write(destination/path.name,forecast)
    path=source/"resource-policy.json"
    resource=json.loads(path.read_text()) if path.exists() else None
    set_resource_policy(resource)
    if source.resolve()!=destination.resolve():
        write(destination/path.name,resource)
    path=source/"historical-gate.json"
    gate=json.loads(path.read_text()) if path.exists() else None
    set_historical_gate(gate)
    if source.resolve()!=destination.resolve():
        write(destination/path.name,gate)
    path=source/"inventory-search.json"
    config=json.loads(path.read_text()) if path.exists() else None
    set_inventory_search(config)
    if source.resolve()!=destination.resolve():
        write(destination/path.name,config)


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
    if not (episode["character"].lower()=="ironclad" and episode["final_act"] and env.done and env.won and all(keys) and episode["act"] == 4 and
            fights[-1]["encounter"] == "The Heart" and fights[-1]["hp_after"] > 0 and
            actions == episode["actions"] and episode["ascension"] == 20 and
            not episode.get("start_cards") and not episode.get("start_thin")):
        raise AssertionError("Run did not meet the A20 Heart victory conditions")
    components = {}
    for name in ("event-table", "choice-table", "combat-head", "late", "synergies", "first-boss-relics", "turn-depth", "combat-preferences", "neow", "draft-context", "late-combat", "late-combat-last-act", "final-weights", "direct-actor", "forecast", "boss-tactics", "recovery", "deck-edits", "scenarios", "route", "opening-search", "trajectory-teacher", "boss-draft", "expert-prior", "curriculum-actor", "inventory-value", "choice-attention", "public-belief", "inventory-search", "historical-menu", "historical-gate"):
        component = destination.parent / (name + ".json")
        if component.exists():
            components[name] = json.loads(component.read_text())
    write(destination, dict(episode=episode, keys=keys, weights=w.tolist(), components=components,
                            native_binary_sha256=sha256(Path(engine.__file__).read_bytes()).hexdigest()))
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
        # Test opposite weight changes on the same seeds so both directions face the same starting conditions.
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
    if args.load:
        configure(args.load.parent, args.out)
        w = np.asarray(json.loads(args.load.read_text())["weights"])
        write(args.out/"policy.json", json.loads(args.load.read_text()))
    else:
        w = train(args)
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
