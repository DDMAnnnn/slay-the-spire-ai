"""Compare plain rounding with an integer-attraction penalty, on unused A20 fight seeds."""
import argparse
import json
from pathlib import Path
import subprocess

import torch

from battle.experience import records
from battle.memory import MemoryBank, observe, point_outcome, save_memory
from battle.memory_training import fit
from .run import describe, read_split, SPLITS


def held_out(exe, folder, name, seed, count, pommel):
    path = folder / f"{name}.jsonl"
    with path.open("wb") as out:
        subprocess.run([str(exe.resolve()), str(seed), str(count), str(int(pommel))],
                       stdout=out, check=True)
    examples = []
    for row in records(path):
        if row["terminal"]:
            continue
        state, action = row["state"], row["action"]
        describe(state)
        e = observe(state, action, row["after"], str(row["fight"]), row["step"])
        if e is None:
            continue
        ordinary = False
        if e.cue.key == ("card", "Strike_R", 0):
            target = state["monsters"][int(action.split()[2])]
            ordinary = (target["block"] == 0 and target["current_hp"] > 6
                        and not any(p["id"] in ("Weak", "Strength") and p["amount"]
                                    for p in state["player"]["powers"])
                        and not any(p["id"] == "Vulnerable" and p["amount"] for p in target["powers"]))
        examples.append((e, ordinary))
    return examples


def compare(model, bank, examples, field):
    forecast = [model.interpret(e.cue, bank)[0].get(field, 0.) for e in examples]
    actual = [e.changes.get(field, 0) for e in examples]
    points = [point_outcome(field, v) for v in forecast]
    n = len(actual)
    return dict(n=n, observed=sum(actual)/n, raw_mean=sum(forecast)/n,
                raw_mae=sum(abs(p-y) for p,y in zip(forecast,actual))/n,
                exact=sum(p==y for p,y in zip(points,actual)),
                point_mae=sum(abs(p-y) for p,y in zip(points,actual))/n,
                distance_to_integer=sum(abs(p-q) for p,q in zip(forecast,points))/n)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("traces", type=Path)
    p.add_argument("collector", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    train, starter, adapt = (read_split(args.traces/(s+".jsonl"), SPLITS[s])
                             for s in ("train","starter-memory","adapt"))
    seen, novel = set(), []
    for e in adapt:
        if e.cue.key[1] == "Pommel Strike" and e.fight not in seen:
            novel.append(e)
            seen.add(e.fight)
    if len(novel) < 16:
        raise ValueError("Insufficient independent Pommel observations")
    starters = held_out(args.collector, args.output, "starter", 8000, 40, False)
    pommels = held_out(args.collector, args.output, "pommel", 9000, 60, True)
    groups = [
        ("ordinary_strike", [e for e,ordinary in starters if ordinary], "target.current_hp", 0),
        ("all_strikes", [e for e,_ in starters if e.cue.key[1]=="Strike_R"], "target.current_hp", 0),
        ("strike_arrivals", [e for e,_ in starters if e.cue.key[1]=="Strike_R"], "hand.arrivals", 0),
        ("defend", [e for e,_ in starters if e.cue.key[1]=="Defend_R"], "player.block", 0),
        ("bash", [e for e,_ in starters if e.cue.key[1]=="Bash"], "target.power.Vulnerable", 0),
    ]
    pommel = [e for e,_ in pommels if e.cue.key[1]=="Pommel Strike"]
    for shots in (0,1,4,16):
        groups.append((f"pommel_draw_{shots}", pommel, "hand.arrivals", shots))
    report = {}
    for name, weight in (("round_only",0.),("integer_loss",0.05)):
        model = fit(train, integer_weight=weight)
        report[name] = {}
        for label, examples, field, shots in groups:
            bank = MemoryBank(starter + novel[:shots])
            report[name][label] = compare(model, bank, examples, field)
        if weight:
            save_memory(args.output/"integer-adapted.pt", model, MemoryBank(starter + novel[:16]))
    (args.output/"results.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf8")
    for k in report["round_only"]:
        print(k, "n, true, raw, raw MAE, integer exact (original | penalty)")
        for branch in report.values():
            r = branch[k]
            print(r["n"], *(round(r[f],3) for f in ("observed","raw_mean","raw_mae")), r["exact"])


if __name__=="__main__":
    main()
