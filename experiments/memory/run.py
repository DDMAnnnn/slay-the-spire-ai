"""Reproduce the isolated starter-deck and Pommel tests, with fresh fights for each split."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import subprocess

import torch

from battle.experience import records, transitions
from battle.memory import MemoryBank, MemoryInterpreter, observe, save_memory, load_memory
from battle.memory_training import fit

# The simulator exports IDs. Supply only the English text and type visible to
# a player for the unupgraded test cards. No card programs enter the interpreter.
DISPLAY = {
    "Strike_R": ("ATTACK", "Deal 6 damage."),
    "Defend_R": ("SKILL", "Gain 5 Block."),
    "Bash": ("ATTACK", "Deal 8 damage. Apply 2 Vulnerable."),
    "Pommel Strike": ("ATTACK", "Deal 9 damage. Draw 1 card."),
    "Slimed": ("STATUS", "Exhaust."),
}
SPLITS = {"train": (1000,96,0), "starter-memory": (2000,16,0), "adapt": (3000,40,1),
          "starter-test": (6000,40,0), "pommel-test": (7000,60,1)}
POMMEL = "Pommel Strike"
METRICS = ("target.current_hp", "player.block", "target.power.Vulnerable", "hand.arrivals")


def describe(state):
    for obj in state["hand"]:
        if not obj.get("upgrades") and not obj.get("description") and obj["id"] in DISPLAY:
            obj["type"],obj["description"] = DISPLAY[obj["id"]]


def read_split(path, spec):
    episodes, fights = [], set()
    for row in records(path):
        fights.add(row["fight"])
        if row["terminal"]:
            continue  # no next player decision to observe
        state = row["state"]
        describe(state)
        item = observe(state,row["action"],row["after"],row["fight"],row["step"])
        if item:
            episodes.append(item)
    start, n, _ = spec
    if fights != set(range(start,start+n)):
        raise ValueError(f"Incomplete or unexpected fights in {path}")
    return episodes


def metrics(interpreter, bank, examples):
    if not examples:
        raise ValueError("No held-out observations")
    totals = defaultdict(float)
    for e in examples:
        pred,_ = interpreter.interpret(e.cue,bank)
        for k in METRICS:
            y = e.changes.get(k,0)
            p = pred.get(k,0)
            totals[k + ".mae"] += abs(p-y)
            totals[k + ".predicted"] += p
            totals[k + ".observed"] += y
            totals[k + ".rounded_exact"] += round(p)==y
    return dict(n=len(examples), fights=len({e.fight for e in examples}),
                **{k:round(v/len(examples),4) for k,v in totals.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder",type=Path)
    parser.add_argument("--collect",type=Path,help="Pinned simulator collector executable")
    parser.add_argument("--recordings",type=Path,help="Additional independent real-game check")
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.folder.mkdir(parents=True,exist_ok=True)
    splits = {}
    for name,spec in SPLITS.items():
        path = args.folder / (name+".jsonl")
        if args.collect:
            with path.open("wb") as stream:
                subprocess.run([str(args.collect.resolve()),*map(str,spec)],stdout=stream,check=True)
        splits[name] = read_split(path,spec)

    training = splits["train"]
    assert all(e.cue.key[1]!=POMMEL for key in ("train","starter-memory") for e in splits[key])
    model = fit(training)
    torch.manual_seed(19)
    cold = MemoryInterpreter(model.fields).eval()
    base = MemoryBank(splits["starter-memory"])
    report = {"counts":{k:dict(Counter(e.cue.key[1] for e in v)) for k,v in splits.items()},
              "starter":{}, "pommel":{}}
    for card_id in ("Strike_R","Defend_R","Bash"):
        tests = [e for e in splits["starter-test"] if e.cue.key[1]==card_id]
        report["starter"][card_id] = {
            name:metrics(m,bank,tests) for name,m,bank in
            (("cold",cold,MemoryBank()),("weights_only",model,MemoryBank()),("starter",model,base))}
    tests = [e for e in splits["pommel-test"] if e.cue.key[1]==POMMEL]
    adaptation = []
    seen_fights = set()
    for e in splits["adapt"]:
        if e.cue.key[1]==POMMEL and e.fight not in seen_fights:
            adaptation.append(e)
            seen_fights.add(e.fight)
    if len(adaptation)<16:
        raise ValueError("Need Pommel observations from at least 16 separate adaptation fights.")
    for n in (0,1,4,16):
        stage = {}
        for name, interpreter, starter in (("cold",cold,[]),("weights_only",model,[]),
                                            ("starter",model,base.episodes())):
            bank = MemoryBank([*starter,*adaptation[:n]])
            stage[name] = metrics(interpreter,bank,tests)
        report["pommel"][n] = stage
    if args.recordings:
        real = []
        for path in sorted(args.recordings.glob("*.jsonl")):
            for i, step in enumerate(transitions(records(path))):
                if step.after_state is None:
                    continue
                describe(step.state)
                e = observe(step.state,step.action,step.after_state,path.stem,i)
                if e:
                    real.append(e)
        report["real"] = {k:metrics(model,base,[e for e in real if e.cue.key[1]==k])
                          for k in sorted({e.cue.key[1] for e in real})}
    # Keep both milestones separately so the zero-shot condition can be replayed.
    save_memory(args.folder/"starter.pt",model,base)
    final = MemoryBank([*base.episodes(),*adaptation[:16]])
    save_memory(args.folder/"adapted.pt",model,final)
    restored,bank = load_memory(args.folder/"adapted.pt")
    assert metrics(restored,bank,tests)==report["pommel"][16]["starter"]
    (args.folder/"results.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf8")
    print("Scored disjoint fights and verified the reloaded checkpoint:",args.folder/"results.json")


if __name__=="__main__":
    main()
