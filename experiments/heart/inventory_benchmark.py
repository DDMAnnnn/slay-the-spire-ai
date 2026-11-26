"""Measure the separate contribution of decks and relics in late standalone fights."""
import argparse
from collections import Counter
import json
from pathlib import Path

import heart_native as engine
import numpy as np
import sts_env as sts

from .expert_draft import DATA_REV,Menus
from .replay import load_model
from .train import write


def winners(root):
    encoder=Menus()
    rows=[]
    seen=set()
    skipped=Counter()
    for group in ("200-rotating-sample","chegs","panacea-ironclad-sample"):
        for f in sorted((root/group).rglob("*.run")):
            r=json.loads(f.read_bytes())
            if (r["character_chosen"]!="IRONCLAD" or r["ascension_level"]!=20 or not r["victory"] or
                not r["damage_taken"] or r["damage_taken"][-1]["enemies"]!="The Heart" or
                any(r.get(k) for k in ("is_daily","chose_seed","is_trial","is_endless"))):
                continue
            seed=str(r["seed_played"])
            if seed in seen:continue
            seen.add(seed)
            deck=[]
            for name in r["master_deck"]:
                key,_,up=name.partition("+")
                if key not in encoder.cards or (up and up!="1"):
                    skipped["unrepresentable card or upgrade"]+=1
                    break
                deck.append((encoder.cards[key]+1)*256+int(bool(up)))
            else:
                if any(x not in encoder.relics for x in r["relics"]) or len(r["relics"])>40 or len(deck)>96:
                    skipped["unrepresentable inventory"]+=1
                    continue
                relics=[encoder.relics[x]+1 for x in r["relics"]]
                rows.append((deck,relics))
    return rows,dict(skipped)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=Path("build/inventory-benchmark"))
    p.add_argument("--threads",type=int,default=6)
    args=p.parse_args()
    args.out.mkdir(exist_ok=True,parents=True)
    human,skips=winners(Path("build/expert-runs/runs"))
    w=load_model("first","build/trajectory-teacher/challenger.json.gz")
    floors=np.load("build/ironclad-curriculum/parent-floors.npy")[:,0]
    rng=np.random.default_rng(165010)
    origins=159000000+rng.choice(np.flatnonzero(floors>=50),size=len(human),replace=False)
    own=engine.bank_inventories(engine.Bank(w,origins.tolist(),50,args.threads))
    profiles=[]
    labels=("native","human_deck","human_relics","human_both","shuffled_human_a","shuffled_human_b","shuffled_native")
    # A random ordering followed by nonzero rotations gives derangements with identical relic marginals.
    order=rng.permutation(len(human))
    for j,index in enumerate(order):
        hd,hr=human[index];od,orr=own[index]
        next_a=order[(j+1)%len(order)]
        next_b=order[(j+len(order)//2)%len(order)]
        profiles.extend([(od,orr),(hd,orr),(od,hr),(hd,hr),
                         (hd,human[next_a][1]),(hd,human[next_b][1]),(od,own[next_a][1])])
    draws=12
    raw=np.asarray(engine.inventory_benchmark(w,profiles,draws,11721,args.threads))
    np.save(args.out/"fights.npy",raw)
    groups=raw.reshape(len(human),len(labels),4,draws,5)
    report=dict(source_commit=DATA_REV,human_inventories=len(human),omitted=skips,draws=draws,
                paired_native_seeds=origins[order].tolist(),practice_seed_offset=11721,results={})
    for i,name in enumerate(labels):
        row={}
        for k,foe in enumerate(("time_eater","awakened_one","donu_deca","shield_spear")):
            x=groups[:,i,k].reshape(-1,5)
            assert np.all(x[:,1]==k)
            row[foe]=dict(fights=len(x),clears=int(x[:,2].sum()),truncated=int(x[:,4].sum()),
                          mean_remaining_hp_including_losses=float(x[:,3].mean()))
        report["results"][name]=row
        print(name,row,flush=True)
    # Resample paired inventory sets, since the same deck fights all four opponents on all draws.
    wins=groups[:,:,:,:,2].mean(axis=(2,3))
    boot=rng.integers(len(wins),size=(10000,len(wins)))
    report["paired_differences"]={}
    for name,i in (("deck",1),("relics",2),("both",3)):
        d=wins[:,i]-wins[:,0]
        report["paired_differences"][name]=dict(mean=float(d.mean()),
            paired_bootstrap_95=np.quantile(d[boot].mean(1),[.025,.975]).tolist())
    for name,d in (("matched_vs_shuffled_human",wins[:,3]-wins[:,4:6].mean(1)),
                   ("matched_vs_shuffled_native",wins[:,0]-wins[:,6])):
        report["paired_differences"][name]=dict(mean=float(d.mean()),
            paired_bootstrap_95=np.quantile(d[boot].mean(1),[.025,.975]).tolist())
    write(args.out/"report.json",report)


if __name__=="__main__":
    main()
