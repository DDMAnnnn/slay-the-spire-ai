"""Learn a combat action value from HP outcomes of paired training rollouts."""
import argparse
import json
from pathlib import Path

import heart_native as engine
import numpy as np
import torch
from torch import nn

from .train import rollouts, statistics, write


class FightValue(nn.Module):
    def __init__(self):
        super().__init__()
        self.card = nn.Embedding(engine.card_count+1, 16)
        self.enemy = nn.Embedding(engine.enemy_count, 16)
        self.features = nn.Linear(22, 16)
        self.readout = nn.Linear(16, 1, bias=False)
        for embedding in (self.card, self.enemy):
            nn.init.normal_(embedding.weight, std=.08)

    def forward(self, rows):
        hidden = self.features(rows[:, 2:-1]) + self.card(rows[:, 0].long()) + self.enemy(rows[:, 1].long())
        return self.readout(hidden.relu()).squeeze(-1)

    def export(self):
        return [p.detach().tolist() for p in (self.card.weight, self.enemy.weight,
                self.features.weight, self.features.bias, self.readout.weight[0])]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("prior", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--episodes", type=int, default=8192)
    p.add_argument("--threads", type=int, default=6)
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(830120)
    args.out.mkdir(parents=True, exist_ok=True)
    policy=json.loads(args.prior.read_text())
    w=np.array(policy["weights"])
    for name, setter in (("event-table",engine.set_event_table),("choice-table",engine.set_choice_table)):
        table=json.loads((args.prior.parent/(name+".json")).read_text())
        setter(table)
        write(args.out/(name+".json"), table)
    def collect(start,n):
        batches=[]
        for offset in range(0,n,128):
            rows=engine.combat_practice(w.tolist(), start+offset,min(128,n-offset),args.threads)
            batches.append(torch.from_numpy(np.array(rows,dtype=np.float32)))
            if offset%2048==0:
                print("collect",start,offset+min(128,n-offset),flush=True)
        return torch.cat(batches)
    data=collect(8000000,args.episodes)
    valid=collect(8300000,384)
    model=FightValue()
    opt=torch.optim.AdamW(model.parameters(),lr=.003)
    loss_fn=nn.SmoothL1Loss(beta=3.)
    best=float("inf")
    print("samples",len(data),len(valid),flush=True)
    for epoch in range(14):
        for idx in torch.randperm(len(data)).split(2048):
            rows=data[idx]
            loss=loss_fn(model(rows),rows[:,-1])
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            error=loss_fn(model(valid),valid[:,-1]).item()
        if error<best:
            best=error
            head=model.export()
        print("epoch",epoch,"validation Huber",error,flush=True)
    best_score=-np.inf
    trials=[]
    for scale in (0.,1.,3.,6.,10.):
        engine.set_head(*head,scale)
        rows=rollouts(w,8500000,4096,args.threads)
        score=(rows[:,0]/56+.5*(rows[:,1]>=2)+5*(rows[:,1]>=3)+30*(rows[:,1]==4)+100*rows[:,4]).mean()
        trials.append(dict(scale=scale,score=float(score),**statistics(rows)))
        if score>best_score:
            write(args.out/"combat-head.json",[*head,scale])
            best_score=score
        print(trials[-1],flush=True)
    policy["fight_value"]=dict(training_examples=len(data),validation_examples=len(valid),huber=best,trials=trials)
    write(args.out/"policy.json",policy)


if __name__=="__main__":
    main()
