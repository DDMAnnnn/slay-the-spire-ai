"""Learn event decisions from named choices logged by human A20 Heart winners."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .heart_curriculum import read_model, save_model
from .human_campfires import archive_rows
from .human_menus import ordinary
from .train import write


# The labels are identifiers. Their offsets match GameContext::chooseEventOption
# and the legal masks in GameAction.cpp. No reward or utility is assigned here.
MENUS = {
    "Back to Basics": ("Ancient Writing", ["Elegance","Simplicity"]),
    "Big Fish": ("Big Fish", ["Banana","Donut","Box"]),
    "Living Wall": ("Living Wall", ["Forget","Change","Grow"]),
    "Winding Halls": ("Winding Halls", ["Embrace Madness","Writhe","Max HP"]),
    "Mysterious Sphere": ("Mysterious Sphere", ["Fight","Ignored"]),
    "Golden Shrine": ("Golden Shrine", ["Pray","Desecrate","Ignored"]),
    "Liars Game": ("The Ssssserpent", ["AGREE","Ignored"]),
    "Drug Dealer": ("Augmenter", ["Obtain J.A.X.","Became Test Subject","Inject Mutagens"]),
    "World of Goop": ("World of Goop", ["Gather Gold","Left Gold"]),
    "Shining Light": ("Shining Light", ["Entered Light","Ignored"]),
    "Ghosts": ("Council of Ghosts", ["Became a Ghost","Ignored"]),
    "The Library": ("The Library", ["Read","Heal"]),
    "SensoryStone": ("Sensory Stone", ["Memory 1","Memory 2","Memory 3"]),
    "Mushrooms": ("Hypnotizing Colored Mushrooms", ["Fought Mushrooms","Healed and dodged fight"]),
    "The Mausoleum": ("The Mausoleum", ["Opened","Ignored"]),
    "MindBloom": ("Mindbloom", ["Fight","Upgrade","Gold","Heal"]),
    "Masked Bandits": ("Masked Bandits", ["Paid Fearfully","Fought Bandits"]),
    "FaceTrader": ("Face Trader", ["Touch","Trade","Leave"]),
    "Duplicator": ("Duplicator", ["Copied","Ignored"]),
    "Transmorgrifier": ("Transmorgrifier", ["Transformed","Ignored"]),
    "Purifier": ("Purifier", ["Purged","Ignored"]),
    "Upgrade Shrine": ("Upgrade Shrine", ["Upgraded","Ignored"]),
    "SecretPortal": ("Secret Portal", ["Entered Portal","Ignored"])
}


def collect():
    src=Path("build/rl-fork/include/constants/Events.h").read_text(encoding="utf8")
    names=re.findall(r'"([^"]+)"',src.split("eventGameNames[]",1)[1].split("};",1)[0])
    seen=set();rows=[];exclusions=Counter();counts=Counter()
    for r in archive_rows():
        if (not ordinary(r) or not r["victory"] or r["floor_reached"]!=57 or not r["damage_taken"]
                or r["damage_taken"][-1]["enemies"]!="The Heart"):continue
        seed=str(r["seed_played"])
        if seed in seen:continue
        seen.add(seed);bucket=int.from_bytes(hashlib.sha256(seed.encode()).digest()[:4],"big")%10
        for e in r["event_choices"]:
            event=e["event_name"];floor=int(e["floor"])
            if event not in MENUS or floor<2:
                exclusions["outside_mapped_menus"]+=1;continue
            native,choices=MENUS[event]
            if e["player_choice"] not in choices:
                exclusions["unrecognized_label"]+=1;continue
            pick=choices.index(e["player_choice"])
            mask=[i<len(choices) for i in range(4)]
            if event=="MindBloom":mask[2 if floor>40 else 3]=False
            if not mask[pick]:
                exclusions["incompatible_menu"]+=1;continue
            hp=r["current_hp_per_floor"][floor-2];mx=r["max_hp_per_floor"][floor-2]
            miss=1.-hp/max(1,mx);gold=r["gold_per_floor"][floor-2]
            features=[1.,miss,miss*miss,(mx-75)/50,min(gold/200,3.),(floor-1)%17/15.,floor>=35]
            rows.append((names.index(native),features,mask,pick,bucket));counts[event]+=1
    return names,rows,{"games":len(seen),"counts":dict(counts),"exclusions":dict(exclusions)}


def main():
    torch.set_num_threads(2)
    out=Path("build/human-events");out.mkdir(exist_ok=True,parents=True)
    names,rows,report=collect()
    events,x,legal,y,split=[torch.tensor(a) for a in zip(*rows)]
    train,val,test=[torch.where(v)[0] for v in (split<8,split==8,split==9)]
    seen=torch.bincount(events[train],minlength=len(names))>=15
    model=read_model("build/route-irl/selected-whole_map.json.gz")
    for contextual,tag in ((False,"frequency"),(True,"context")):
        torch.manual_seed(258113)
        params=nn.Parameter(torch.zeros(len(names),4,7))
        opt=torch.optim.AdamW([params],lr=.028,weight_decay=.035)
        def logits(ids):
            z=params[events[ids],:,0]
            if contextual:z=z+(params[events[ids],:,1:]*x[ids,None,1:]).sum(2)
            return z.masked_fill(~legal[ids],-1e7)
        @torch.no_grad()
        def grade(ids):
            z=logits(ids);return dict(n=len(ids),nll=float(F.cross_entropy(z,y[ids])),
                                      accuracy=float((z.argmax(1)==y[ids]).float().mean()))
        best=float("inf");history=[]
        for ep in range(1,33):
            for ids in train[torch.randperm(len(train))].split(256):
                opt.zero_grad();F.cross_entropy(logits(ids),y[ids]).backward();opt.step()
            if ep in (1,2,4,8,16,32):
                v=grade(val);history.append(dict(epoch=ep,**v))
                if v["nll"]<best:best=v["nll"];selected=ep;saved=params.detach().clone()
        with torch.no_grad():params.copy_(saved)
        report[tag]=dict(training=len(train),validation=history,epoch=selected,test=grade(test))
        save_model(out/f"{tag}.json.gz",{**model,"human_events":[params.detach().tolist(),seen.tolist(),1.]})
    write(out/"fit.json",report);print(report["frequency"]["test"],report["context"]["test"],flush=True)


if __name__=="__main__":
    main()
