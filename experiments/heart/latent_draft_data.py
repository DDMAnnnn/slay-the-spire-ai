"""Attach a training-only final deck to each human menu, checking historical order and splits."""
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from .human_menus import PrefixReader, STARTERS, TOKENS, ordinary


def main():
    out=Path("build/latent-draft");out.mkdir(parents=True,exist_ok=True)
    reader=PrefixReader();groups=defaultdict(list);seen=set()
    with gzip.open("build/historical-human/ironclad.jsonl.gz","rt",encoding="utf8") as fp:
        for r in map(json.loads,fp):
            seed=str(r["seed_played"])
            if (not ordinary(r) or seed in seen or not r["victory"] or r["floor_reached"]!=57 or
                    not r["damage_taken"] or r["damage_taken"][-1]["enemies"]!="The Heart"):
                continue
            seen.add(seed)
            split=int.from_bytes(hashlib.sha256(seed.encode()).digest()[:4],"big")%10
            group="test" if split==9 else "validation" if split==8 else "train"
            deck=np.zeros(TOKENS,np.uint8)
            for name in r["master_deck"]:
                base=name.partition("+")[0];token=reader.cards.get(base)
                if base not in STARTERS and token is not None:deck[token]=min(3,int(deck[token])+1)
            menus=list(reader.run(r))
            groups[group].append((deck,menus))
    for group,records in groups.items():
        decks=np.stack([v[0] for v in records]);owner=[]
        with np.load(f"build/human-strategy/{group}.npz") as data:
            for gi,(_,menus) in enumerate(records):
                owner.extend([gi]*len(menus))
            if len(owner)!=len(data["pick"]):raise ValueError("human menu count differs")
            # Check all labels, contexts, and menu identities before attaching future-deck targets.
            ctx_all=data["context"];choices=data["options"];picks=data["pick"];i=0
            for _,menus in records:
                for ctx,act,kind,options,choice in menus:
                    if picks[i]!=choice or not np.array_equal(ctx_all[i],ctx) or not np.array_equal(choices[i,:len(options)],options):
                        raise ValueError(f"unaligned historical menu {group} {i}")
                    i+=1
        np.savez_compressed(out/f"{group}.npz",decks=decks,owner=np.asarray(owner))
        print(group,len(decks),len(owner),flush=True)


if __name__=="__main__":main()
