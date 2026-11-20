"""Does printed card text help choose an unseen composition? A one-action combat bandit."""
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random

import torch
from torch.nn import functional as F

from battle.__main__ import load, save
from battle.policy import CombatPolicy, action_indices, scalars


HOLDOUT = (10, 2)
KNOWN = [(amount, hits) for amount in (4, 6, 8, 10, 12, 14, 18, 22)
         for hits in (1, 2) if (amount, hits) != HOLDOUT]
OPTIONS = ("play 0 0", "play 1 0", "end")


def identity(rule, alias):
    return hashlib.sha256(f"fixture/{rule}/{alias}".encode()).hexdigest()[:16]


def card(rule, alias):
    amount, hits = rule
    return {"id": identity(rule, alias), "type": "ATTACK", "cost": 1, "upgrades": 0,
            "target": "ENEMY", "description": f"Deal {amount} damage" + (" twice." if hits == 2 else ".")}


@dataclass
class Fight:
    state: dict
    winner: int
    group: str


def lethal_decision(rng, rules, strength, group, aliases):
    # These mechanics belong only to the fixture. Neither damage nor hits is
    # provided as a separate card feature, training label or action bonus.
    totals = [(amount + strength)*hits for amount, hits in rules]
    if totals[0] == totals[1]:
        return None
    block = rng.choice((0, 1, 3))
    hp = rng.randint(min(totals) + 1, max(totals)) - block
    assert hp > 0
    hand = [card(rule, alias) for rule, alias in zip(rules, aliases)]
    powers = [{"id":"Strength","amount":strength}] if strength else []
    state = {"player":{"current_hp":5,"max_hp":80,"block":0,"powers":powers,"potions":[]},
             "turn":1,"energy":1,"hand":hand,"draw":[],"discard":[],"exhaust":[],
             "monsters":[{"id":"fixture_foe","current_hp":hp,"max_hp":100,"block":block,"powers":[],
                          "intent":"ATTACK","damage":8,"hits":1}]}
    return Fight(state, totals.index(max(totals)), group)


def familiar(seed, count):
    rng = random.Random(seed)
    fights = []
    while len(fights) < count:
        trial = lethal_decision(rng, rng.sample(KNOWN,2), rng.choice((0,2,4)), "familiar",
                                [rng.randrange(4),rng.randrange(4)])
        if trial:
            fights.append(trial)
    return fights


def novel(seed, pairs_per_group):
    rng = random.Random(seed)
    fixed, swapped = [], []
    # Balance whether the held-out card should be played. The switch case tests
    # a harder distinction: 22 once wins at 0 Strength; 10 twice wins at 4.
    for group, opponent, strengths in (("repetition",(10,1),(0,2,4)),
                                       ("amount",(12,2),(0,2,4)),
                                       ("tradeoff",(22,1),(0,4))):
        for i in range(pairs_per_group):
            rules = [HOLDOUT, opponent]
            aliases = [0,rng.randrange(4)]
            if i%2:
                rules.reverse()
                aliases.reverse()
            trial = lethal_decision(rng,rules,strengths[(i//2)%len(strengths)],group,aliases)
            fixed.append(trial)
            # Two hypothetical worlds, identical IDs, state and slot positions.
            # Exchange only their descriptions and therefore their actual effects.
            cards = trial.state["hand"]
            for flip in (False,True):
                hand = [dict(cards[j], id=identity("new-world",j),
                             description=cards[1-j if flip else j]["description"]) for j in (0,1)]
                swapped.append(Fight(dict(trial.state,hand=hand), 1-trial.winner if flip else trial.winner,group))
    return fixed, swapped


class Batch:
    """Deduplicate identical public rows only to batch the normal policy computation."""
    def __init__(self, model, fights):
        rows, mapping, states = [], {}, []
        for fight in fights:
            objects, addresses = model.read_state(fight.state)
            assert [action_indices(o,addresses) for o in OPTIONS] == [(0,1,3),(0,2,3),(2,-1,-1)]
            state = []
            for obj in objects:
                key = (tuple(obj.words),tuple(scalars(obj)))
                if key not in mapping:
                    mapping[key] = len(rows)
                    rows.append(obj)
                state.append(mapping[key])
            states.append(state)
        self.rows = rows
        self.states = torch.tensor(states)
        self.winners = torch.tensor([f.winner for f in fights])

    def scores(self, model, ids):
        selected = self.states[ids]
        unique, reverse = torch.unique(selected, return_inverse=True)
        vectors = model.embed_rows([self.rows[i] for i in unique.tolist()])
        context = model.battle(vectors[reverse.reshape_as(selected)])
        batch = len(ids)
        absent = model.absent.expand(batch,1,-1)
        sources = torch.cat((context[:,1:3],absent),dim=1)
        targets = torch.cat((context[:,3:4].expand(-1,2,-1),absent),dim=1)
        kinds = torch.tensor([0,0,2]).expand(batch,-1)
        scores,_ = model.score_rows(context[:,:1].expand(-1,3,-1),sources,targets,kinds)
        return scores


@torch.no_grad()
def evaluate(model, fights):
    batch = Batch(model, fights)
    indices = torch.arange(len(fights))
    choices = torch.cat([batch.scores(model,chunk).argmax(-1) for chunk in indices.split(128)])
    # Check the same action logits through the actual game-policy entry point.
    torch.testing.assert_close(batch.scores(model,indices[:1])[0],model(fights[0].state, OPTIONS)[0],
                               atol=2e-5,rtol=2e-5)
    correct = choices == batch.winners
    groups = {"all":indices}
    for group in {f.group for f in fights}:
        groups[group] = [i for i,f in enumerate(fights) if f.group==group]
    return {g:{"wins":int(correct[ix].sum()),"fights":len(ix)} for g,ix in groups.items()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("output",type=Path)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    train = familiar(12000,4096)
    familiar_test = familiar(13000,384)
    held,paired = novel(14000,128)
    results = []
    for seed in (19,23,41):
        # Both arms receive exactly the same logged action and terminal win/loss.
        generator = torch.Generator().manual_seed(seed+500)
        logged_actions = torch.randint(3,(len(train),),generator=generator)
        schedule = torch.randint(len(train),(480,48),generator=generator)
        for mode in ("id","id_text"):
            torch.manual_seed(seed)
            model = CombatPolicy(width=32,heads=4,card_input=mode)
            optimizer = torch.optim.AdamW(model.parameters(),lr=.001)
            batch = Batch(model,train)
            for indices in schedule:
                chosen = logged_actions[indices]
                outcome = (chosen == batch.winners[indices]).float()
                scores = batch.scores(model,indices).gather(1,chosen[:,None]).squeeze(1)
                loss = F.binary_cross_entropy_with_logits(scores,outcome)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step()
            path=args.output/f"{mode}-{seed}.pt"
            save(model,path)
            restored=load(path)
            row=dict(seed=seed,mode=mode,familiar=evaluate(restored,familiar_test),
                     held=evaluate(restored,held),paired=evaluate(restored,paired))
            results.append(row)
            print(seed,mode,*(row[k]["all"] for k in ("familiar","held","paired")),flush=True)
    (args.output/"results.json").write_text(json.dumps(results,indent=2)+"\n",encoding="utf8")


if __name__=="__main__":
    main()
