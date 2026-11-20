"""Evaluate one Pommel action twice from a matched simulator state, with and without No Draw."""
import argparse
import copy
import json
from pathlib import Path

import torch

from battle.experience import records
from battle.memory import MemoryBank, load_memory, observe, point_outcome
from .run import describe


def pairs(path):
    result = []
    seen = set()
    for row in records(path):
        fight = row["fight"]
        if fight in seen:
            raise ValueError(f"Repeated source fight: {fight}")
        seen.add(fight)
        first = row["allowed"]["state"]
        expected = copy.deepcopy(first)
        expected["player"]["powers"].append({"id":"No Draw","amount":1})
        if expected != row["blocked"]["state"]:
            raise ValueError(f"Unmatched public observations for fight {fight}")
        pair = []
        for branch in ("allowed","blocked"):
            item = row[branch]
            describe(item["state"])
            e = observe(item["state"], row["action"], item["after"], str(fight), row["step"])
            if e is None or e.cue.key[1] != "Pommel Strike":
                raise ValueError(f"Invalid action window in fight {fight}")
            pair.append(e)
        # Check measured simulator outcomes; never substitute these labels at inference.
        if [e.changes.get("hand.arrivals",0) for e in pair] != [1,0]:
            raise ValueError(f"Unexpected observed draws in fight {fight}")
        result.append(pair)
    return result


def evaluate(model, bank, cases, *, writable):
    size_before = len(bank.seen)
    totals = [0.,0.]
    correct = [0,0]
    both = 0
    for index,pair in enumerate(cases):
        # Predict BOTH before revealing either. Write only one realized branch.
        predictions = [model.interpret(e.cue,bank)[0]["hand.arrivals"] for e in pair]
        success = []
        for branch,(pred,e) in enumerate(zip(predictions,pair)):
            totals[branch] += pred
            hit = point_outcome("hand.arrivals",pred) == e.changes.get("hand.arrivals",0)
            correct[branch] += hit
            success.append(hit)
        both += all(success)
        if writable:
            bank.remember(pair[index%2])
    n = len(cases)
    assert len(bank.seen) == size_before + (n if writable else 0)
    return dict(pairs=n, mean_allowed=totals[0]/n, mean_blocked=totals[1]/n,
                mean_paired_difference=(totals[0]-totals[1])/n, allowed_correct=correct[0],
                blocked_correct=correct[1], both_correct=both,
                episodes_before=size_before,episodes_after=len(bank.seen))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint",type=Path)
    p.add_argument("folder",type=Path)
    args=p.parse_args()
    torch.set_num_threads(1)
    model,starter=load_memory(args.checkpoint)
    train=pairs(args.folder/"adapt.jsonl")
    test=pairs(args.folder/"test.jsonl")
    if len(train)<12 or not test:
        raise ValueError("Need 12 adaptation pairs and at least one held-out pair.")
    ids = lambda cases: {pair[0].fight for pair in cases}
    if ids(train) & ids(test) or (ids(train)|ids(test)) & {e.fight for e in starter.episodes()}:
        raise ValueError("Overlapping memory and evaluation fights.")
    if any(key[1]=="Pommel Strike" for key in starter.by_key):
        raise ValueError("Use the starter checkpoint before Pommel adaptation.")
    older=starter.episodes()
    normal=[pair[0] for pair in train[:12]]
    mixed=[pair[i%2] for i,pair in enumerate(train[:12])]
    output = {}
    for name,episodes,writable in (
        ("weights_only",[],False),("starter",older,False),
        ("allowed_only",older+normal,False),("balanced",older+mixed,False),
        ("online_from_starter",older,True),
    ):
        output[name]=evaluate(model,MemoryBank(episodes),test,writable=writable)
    text=json.dumps(output,indent=2)
    (args.folder/"results.json").write_text(text+"\n",encoding="utf8")
    print(text)


if __name__=="__main__":
    main()
