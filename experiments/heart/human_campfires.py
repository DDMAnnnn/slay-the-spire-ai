"""Reconstruct human campfire choices from floor prefixes and fit masked choice models."""
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re
import zipfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .expert_draft import Menus
from .heart_curriculum import read_model, save_model
from .human_menus import DIGEST, ordinary
from .resource_imitation import ResourcePolicy
from .train import write


REST = {"REST": 0, "SMITH": 1, "RECALL": 2, "LIFT": 3, "PURGE": 4, "DIG": 5}
FEATURES = (0,1,2,3,4,5,6,7,8,9,17,18)


def archive_rows():
    path = Path("build/historical-human/A20runs.zip")
    if hashlib.sha256(path.read_bytes()).hexdigest() != DIGEST:
        raise ValueError("human archive checksum differs")
    with zipfile.ZipFile(path) as z, z.open("A20runs.json") as fp:
        columns = json.load(fp)
    for idx, who in columns["character_chosen"].items():
        if who == "IRONCLAD":
            yield {k: entries.get(idx) for k,entries in columns.items()}


def collect():
    menus = Menus()
    src = Path("build/rl-fork/include/constants/Cards.h").read_text(encoding="utf8")
    types = re.findall(r"CardType::([A-Z]+)", src.split("cardTypes[]",1)[1].split("};",1)[0])
    assert len(types) == len(menus.cards)
    seen = set()
    rows = [[], []]
    diagnostic = Counter()
    for r in archive_rows():
        if (not ordinary(r) or not r["victory"] or not r["damage_taken"] or
                r["damage_taken"][-1]["enemies"] != "The Heart" or r["floor_reached"] != 57):
            continue
        seed = str(r["seed_played"])
        if seed in seen:
            continue
        seen.add(seed)
        bucket = int.from_bytes(hashlib.sha256(seed.encode()).digest()[:4],"big") % 10
        deck = Counter()
        if r["neow_bonus"] in ("THREE_ENEMY_KILL","TWO_FIFTY_GOLD","HUNDRED_GOLD","TEN_PERCENT_HP_BONUS",
                               "TWENTY_PERCENT_HP_BONUS","THREE_SMALL_POTIONS","THREE_CARDS","THREE_RARE_CARDS"):
            deck.update({"Strike_R":5, "Defend_R":4, "Bash":1})
        relics = set()
        red, lifts = False, 0
        changes, fires = defaultdict(list), defaultdict(list)
        for c in r["card_choices"]:
            changes[int(c["floor"])].append(("add", c["picked"]))
        for c in r["relics_obtained"]:
            changes[int(c["floor"])].append(("relic",c["key"]))
        for i,c in enumerate(r["boss_relics"]):
            if c.get("picked"):changes[17*(i+1)].append(("relic",c["picked"]))
        for item,floor in zip(r["items_purchased"],r["item_purchase_floors"]):
            changes[int(floor)].append(("relic" if item in menus.relics else "add",item))
        for item,floor in zip(r["items_purged"],r["items_purged_floors"]):
            changes[int(floor)].append(("remove",item))
        for e in r["event_choices"]:
            for field,op in (("relics_lost","lost"),("relics_obtained","relic"),("cards_removed","remove"),
                             ("cards_transformed","remove"),("cards_upgraded","remove"),("cards_obtained","add")):
                changes[int(e["floor"])].extend((op,v) for v in e.get(field,[]))
        for e in r["campfire_choices"]:
            fires[int(e["floor"])].append(e)
        for floor in sorted(set(changes) | set(fires)):
            for e in fires[floor]:
                choice = e["key"]
                if choice not in REST or floor < 2:
                    continue
                act = min(4,1+(floor-1)//17)
                hp = r["current_hp_per_floor"][floor-2]
                maxhp = r["max_hp_per_floor"][floor-2]
                gold = r["gold_per_floor"][floor-2]
                miss = 1-hp/max(1,maxhp)
                local = (floor-1)%17
                x = [1.,miss,miss*miss,(maxhp-75)/50,min(gold/200,3.),act==2,act>=3,
                     local/15,local>=13,red,miss*(local>=13),act>=3 and not red]
                options = [i for i,ok in enumerate(("Coffee Dripper" not in relics, "Fusion Hammer" not in relics,
                           not red,"Girya" in relics and lifts<3,"Peace Pipe" in relics,"Shovel" in relics)) if ok]
                known = sorted(menus.cards[c] for c,n in deck.items() if n>0)
                target = str(e.get("data","")).split("+")[0]
                # A campfire-entry heal lies between the previous floor's HP and this decision.
                if "Eternal Feather" not in relics:
                    if REST[choice] in options and len(options)>1:
                        rows[0].append((x,options,REST[choice],bucket))
                    else:
                        diagnostic["forced_or_unresolved"]+=1
                    if choice=="SMITH" and menus.cards.get(target,-1) in known and len(known)>1:
                        rows[1].append((x,known,menus.cards[target],bucket))
                else:
                    diagnostic["entry_heal_excluded"]+=1
                if choice in ("SMITH","PURGE") and "+" not in str(e.get("data","")):
                    deck[target]=max(0,deck[target]-1)
                red |= choice=="RECALL"
                lifts += choice=="LIFT"
            for op,item in changes[floor]:
                if op=="relic":
                    if item in relics:continue
                    relics.add(item)
                    if item in ("Pandora's Box","Astrolabe","Empty Cage","War Paint","Whetstone","Tiny House"):
                        deck.clear()
                elif op=="lost":
                    relics.discard(item)
                elif op=="remove":
                    if "+" not in item:deck[item]=max(0,deck[item]-1)
                elif "+" not in item and item in menus.cards:
                    t=types[menus.cards[item]]
                    egg={"ATTACK":"Molten Egg 2","POWER":"Frozen Egg 2","SKILL":"Toxic Egg 2"}.get(t)
                    if egg is not None and egg not in relics:deck[item]+=1
    return rows,dict(verified_archive=DIGEST,unique_winning_seeds=len(seen),exclusions=diagnostic)


def fit(rows, size):
    width=max(len(r[1]) for r in rows)
    options=torch.zeros((len(rows),width),dtype=torch.long)
    legal=torch.zeros_like(options,dtype=torch.bool)
    for i,(_,menu,_,_) in enumerate(rows):
        options[i,:len(menu)]=torch.tensor(menu);legal[i,:len(menu)]=True
    labels=torch.tensor([menu.index(target) for _,menu,target,_ in rows])
    x=torch.tensor([r[0] for r in rows],dtype=torch.float32)
    bucket=torch.tensor([r[3] for r in rows])
    tr,va,te=[torch.where(c)[0] for c in (bucket<8,bucket==8,bucket==9)]
    weights=nn.Parameter(torch.zeros(size,len(FEATURES)))
    opt=torch.optim.AdamW([weights],lr=.024,weight_decay=.035)
    def logits(ids):
        return (weights[options[ids]]*x[ids,None,:]).sum(2).masked_fill(~legal[ids],-1e8)
    @torch.no_grad()
    def grade(ids):
        z=logits(ids)
        return dict(n=len(ids),nll=float(F.cross_entropy(z,labels[ids])),
                    accuracy=float((z.argmax(1)==labels[ids]).float().mean()),
                    chance=float(legal[ids].sum(1).float().reciprocal().mean()))
    best=float("inf");history=[]
    for epoch in range(1,33):
        for ids in tr[torch.randperm(len(tr))].split(256):
            loss=F.cross_entropy(logits(ids),labels[ids]);opt.zero_grad();loss.backward();opt.step()
        if epoch in (1,2,4,8,16,32):
            v=grade(va);history.append(dict(epoch=epoch,**v))
            if v["nll"]<best:
                best=v["nll"];saved=weights.detach().clone();selected=epoch
    with torch.no_grad():weights.copy_(saved)
    return saved,dict(training=len(tr),validation=history,epoch=selected,test=grade(te))


def main():
    out=Path("build/human-campfires");out.mkdir(exist_ok=True,parents=True)
    torch.set_num_threads(2);torch.manual_seed(256202)
    rows,report=collect();sizes=json.loads(Path("build/resource-policy/sizes.json").read_text())
    net=ResourcePolicy(9+sum(sizes[k] for k in ("cards","relics","potions")))
    with torch.no_grad():
        for p in net.parameters():p.zero_()
    for kind,name,examples,size,offset in ((0,"camp",rows[0],6,2),(2,"smith",rows[1],sizes["cards"],9)):
        params,report[name]=fit(examples,size)
        with torch.no_grad():
            for i,feature in enumerate(FEATURES):net.item[kind,offset:offset+size,feature]=params[:,i]
    m=read_model("build/route-irl/selected-whole_map.json.gz")
    save_model(out/"human.json.gz",{**m,"resource_policy":[*net.export(),[1.,0.,1.,0.]]})
    write(out/"fit.json",report);print({k:report[k]["test"] for k in ("camp","smith")},flush=True)


if __name__=="__main__":
    main()
