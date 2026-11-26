"""Train drafting and within-fight tactics together, then compare fixed epoch checkpoints."""
import argparse
from pathlib import Path

import heart_native as engine
import numpy as np
import torch
from torch import nn

from .heart_curriculum import read_model, save_model
from .replay import apply_model, heart_encounters
from .train import statistics, write


def collect(model,menu,start,n,limit,threads,card_trials):
    w=apply_model("first",model)
    full=np.asarray(engine.batch(w,start,n,threads))
    seeds=start+np.flatnonzero(full[:,0]>=32)[:limit]
    inputs,targets,cards=[],[],[]
    for offset in range(0,len(seeds),128):
        engine.set_boss_tactics(menu[1])
        bank=engine.Bank(w,seeds[offset:offset+128].tolist(),32,threads)
        if card_trials:
            cards.extend(bank.recovery_card_trials(w,start+offset,24,threads))
        for i,steps in enumerate((0,5,12,22)):
            engine.set_boss_tactics(menu[(offset//128+i)%len(menu)])
            phase=bank.recovery_phase(w,steps,threads)
            if not len(phase):
                continue
            inputs.extend(phase.recovery_features(w))
            labels=[]
            for tactics in menu:
                engine.set_boss_tactics(tactics)
                r=np.asarray(phase.boss_results(w,threads))
                if np.any(r[:,0]<0):
                    raise RuntimeError("missing boss in tactical rollout")
                labels.append(r[:,1]+.2*(1-r[:,2])+.05*r[:,3])
            targets.extend(np.stack(labels,axis=1))
    return np.asarray(inputs,np.float32),np.asarray(targets,np.float32),np.asarray(cards),len(seeds)


def draft_effects(rows):
    output=np.zeros((engine.card_count,3))
    for card in np.unique(rows[:,0]):
        subset=rows[rows[:,0]==card]
        x=np.column_stack([np.ones(len(subset)),subset[:,1:3]])
        effect=subset[:,4]-subset[:,3]
        output[card]=np.linalg.solve(x.T@x+np.diag([50.,120.,120.]),x.T@effect)
    return output


def train_checkpoints(train,valid,menu,draft,parent):
    torch.manual_seed(10391)
    x,y=map(torch.from_numpy,train)
    vx,vy=valid
    # A hard position penalizes every tactic. Center each matched group so the
    # optimizer spends capacity on the differences its choices can influence.
    target=y-y.mean(1,keepdim=True)
    mean=x.mean(0)
    scale=x.std(0).clamp_min(.15)
    normalized=(x-mean)/scale
    net=nn.Sequential(nn.Linear(40,32),nn.ReLU(),nn.Linear(32,16))
    opt=torch.optim.AdamW(net.parameters(),lr=.0015)
    for epoch in range(1,129):
        for batch in torch.randperm(len(x)).split(192):
            pred=net(normalized[batch])
            loss=(pred-target[batch]).square().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch in (1,4,16,64,128):
            with torch.no_grad():
                values=net((torch.from_numpy(vx)-mean)/scale)
                index=values.argmax(1).numpy()
                gain=(vy[np.arange(len(vy)),index]-vy[:,1]).mean()
                first=net[0].weight/scale
                bias=net[0].bias-first@mean
                params=[menu,first.tolist(),bias.tolist(),net[2].weight.tolist(),net[2].bias.tolist(),draft.tolist()]
                yield epoch,{**parent,"recovery":params},float(gain)


def panel(model,start,n,threads):
    w=apply_model("first",model)
    rows=np.asarray(engine.batch(w,start,n,threads))
    return dict(**statistics(rows),heart_encounter_seeds=heart_encounters(w,start,rows)),rows


def split_models(parent,combined):
    params=combined["recovery"]
    no_draft=np.zeros_like(params[-1]).tolist()
    prior_menu=[parent["boss_tactics"] for _ in params[0]]
    variants={"combat_only":[*params[:-1],no_draft],"draft_only":[prior_menu,*params[1:]]}
    return {name:{**combined,"recovery":weights} for name,weights in variants.items()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=Path("build/recovery"))
    p.add_argument("--threads",type=int,default=10)
    args=p.parse_args()
    args.out.mkdir(exist_ok=True,parents=True)
    torch.set_num_threads(2)
    parent=read_model("build/boss-tactics/challenger.json.gz")
    specialist=np.array(parent["boss_tactics"])
    rng=np.random.default_rng(10823)
    menu=rng.normal(0,22,(16,3,18))
    menu[::2]+=specialist
    menu[0]=0
    menu[1]=specialist
    menu=menu.tolist()
    x,y,cards,n=collect(parent,menu,108000000,16384,3072,args.threads,True)
    vx,vy,_,vn=collect(parent,menu,109000000,4096,768,args.threads,False)
    draft=65*draft_effects(cards)
    report=dict(training_start=108000000,validation_position_start=109000000,
                validation_game_start=110000000,test_start=111000000,training_boss_rooms=n,
                training_positions=len(x),card_trials=len(cards),validation_rooms=vn,
                validation_positions=len(vx),epoch_validation={})
    print("training positions, paired card trials",len(x),len(cards),flush=True)
    models={}
    for epoch,model,gain in train_checkpoints((x,y),(vx,vy),menu,draft,parent):
        models[epoch]=model
        stats,_=panel(model,110000000,4096,args.threads)
        stats["matched_tactical_gain"]=gain
        report["epoch_validation"][epoch]=stats
        print(epoch,stats,flush=True)
    # Declare the ordering before the test. Early stopping for the full policy
    # uses fresh games; only the winning checkpoint moves on to final evaluation.
    def rank(epoch):
        s=report["epoch_validation"][epoch]
        return len(s["heart_encounter_seeds"]),s["act4"],s["act3"],s["matched_tactical_gain"]
    best=max(models,key=rank)
    report["selected_epoch"]=best
    save_model(args.out/"challenger.json.gz",models[best])
    report["test"]={}
    results=[]
    for name,model in (("previous_specialist",parent),("combined",models[best])):
        summary,rows=panel(model,111000000,32768,args.threads)
        report["test"][name]=summary
        results.append(rows)
        print(name,summary,flush=True)
    a,b=results
    report["gained_act3"]=int(((a[:,1]<3)&(b[:,1]>=3)).sum())
    report["lost_act3"]=int(((a[:,1]>=3)&(b[:,1]<3)).sum())
    if (b[:,1]>=3).sum()>1.1*(a[:,1]>=3).sum():
        report["exploratory_components"]={}
        for name,variant in split_models(parent,models[best]).items():
            save_model(args.out/(name.replace("_","-")+".json.gz"),variant)
            report["exploratory_components"][name]=panel(variant,111000000,32768,args.threads)[0]
    write(args.out/"report.json",report)


if __name__=="__main__":
    main()
