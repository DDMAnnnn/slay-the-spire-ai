"""Replay verified public games in their own runtime and serve matched combat starts to the learner."""
import argparse
import base64
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from public_demo_export import combat_index, digest, public_row


def send(value):
    sys.stdout.write(json.dumps(value,separators=(",",":"))+"\n")
    sys.stdout.flush()


def choose(root,tiny):
    with gzip.open(root/"replays.jsonl.gz","rt") as src:
        games=[r for r in map(json.loads,src) if not r["error"] and r["public_information"] and r["original_audit"]["passed"]]
    pools=defaultdict(list)
    for gi,r in enumerate(games):
        bucket=int.from_bytes(hashlib.sha256(str(r["seed"]).encode()).digest()[:4],"big")%10
        split="train" if bucket<8 else "validation" if bucket==8 else "test"
        for index,v in enumerate(r["fights"]):
            pools[split,v["act"]].append((gi,index))
    randomizer=random.Random(23911)
    plans={"train":(100,128,128,64),"validation":(48,48,48,24),"test":(56,56,56,32)}
    picks=defaultdict(dict)
    for (split,act),group in sorted(pools.items()):
        randomizer.shuffle(group)
        wanted=2 if tiny else plans[split][act-1]
        # Distinct games within an act while enough games exist.
        first={};rest=[]
        for g,i in group:
            if g not in first:first[g]=i
            else:rest.append((g,i))
        for g,i in (list(first.items())+rest)[:wanted]:picks[g][i]=split
    return games,picks


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--tiny",action="store_true")
    args=p.parse_args();root=Path("build/public-expert-traces")
    sys.path[:0]=[str((root/"agent"/k).resolve()) for k in ("native","runtime")]
    import slaythespire as sts
    from sts_rl_env import SpireEnv
    layout=json.loads(Path("build/public-expert-intent/layout.json").read_text())
    games,plans=choose(root,args.tiny)
    banks=defaultdict(list)
    for gi,which in sorted(plans.items()):
        r=games[gi];env=SpireEnv(sts,sts.CharacterClass.IRONCLAD,r["seed"],20,"snapshot")
        if digest(env.snapshot())!=r["start_state_hash"]:raise ValueError("mismatched initial state")
        index=-1;last=None
        for d,b in r["trace"]:
            transition=env._native.step(sts.EnvironmentAction(sts.ActionDomain(d),b))
            state=env.snapshot()
            key=(state["act"],state["floor"],state["decision"]["encounter"]) if state.get("combat") else None
            entered=key is not None and key!=last
            last=key
            if entered:
                index+=1
                if index not in which:continue
                f=r["fights"][index]
                if (state["act"],state["floor"],state["decision"]["encounter"])!=(f["act"],f["floor"],f["encounter"]):
                    raise ValueError(f"unmatched expert battle {r['seed']} {index}")
                banks[which[index]].append((env.clone(),r["seed"],index,f))
        if index+1!=len(r["fights"]) or digest(env.snapshot())!=r["terminal_state_hash"]:
            raise ValueError(f"expert battle reconstruction failed for {gi}: counted {index+1}, expected {len(r['fights'])}, terminal hash matches {digest(env.snapshot())==r['terminal_state_hash']}")
    for k in banks:random.Random(23977).shuffle(banks[k])
    send(dict(banks={k:len(v) for k,v in banks.items()}))
    for command in sys.stdin:
        task=json.loads(command)
        if "stop" in task:return
        pool=banks[task["split"]]
        if "subset" in task:
            pool=random.Random(task["subset_seed"]).sample(pool,min(len(pool),task["subset"]))
        pool=[item for item in pool for _ in range(task.get("repeats",1))]
        train=task.get("labels",False)
        rng=random.Random(239993)
        summaries=[]
        for offset in range(0,len(pool),32):
            cohort=pool[offset:offset+32]
            games=[e.clone() for e,*_ in cohort]
            entries=[dict(seed=str(seed),index=ix,act=f["act"],ref_win=f["survived"],ref_hp=f["exit_hp"],
                          max_hp=e.snapshot()["max_hp"],steps=0,followups=0,labels=0,capped=False) for e,seed,ix,f in cohort]
            active=set(range(len(cohort)))
            while active:
                packet=[];need=[];actions={}
                for i in sorted(active):
                    env=games[i];state=env.snapshot();legal=env._native.legal_actions()
                    e=entries[i]
                    index={combat_index(a.bits):a for a in legal if int(a.domain)==1 and combat_index(a.bits) is not None}
                    if 80 in index and len(index)>1:
                        if "search_budget" in task and all(int(a.domain)==1 and combat_index(a.bits) is not None for a in legal):
                            # Fixed by bank index and move count, independently of the real game seed.
                            particle_rng=random.Random(740387*(offset+i+1)+39883*e["steps"])
                            seed=particle_rng.getrandbits(53)
                            if task.get("search_mode",-1)<0:
                                bits=env.public_combat_search_action(task["search_budget"],4,seed)["action"].bits
                            else:
                                bits=env._native.public_belief_combat_search_action(task["search_budget"],4,seed,task["search_mode"],True).action.bits
                            move=combat_index(bits)
                            if move not in index:raise ValueError("search chose an unavailable move")
                            actions[i]=index[move]
                            e["labels"]+=1
                            continue
                        label=255
                        if train and rng.random()<.55:
                            strength=192 if e["act"]>=3 else 96
                            result=env.public_combat_search_action(strength,4,rng.randrange(1<<45))
                            pred=combat_index(result["action"].bits)
                            if pred in index:label=pred;e["labels"]+=1
                        o=public_row(state,set(index),layout)
                        packet.append(o.tobytes()+bytes(int(j in index) for j in range(81))+bytes([label]))
                        need.append((i,index))
                    else:
                        if not legal:raise RuntimeError("empty combat menu")
                        actions[i]=min(legal,key=lambda a:(int(a.domain),int(a.bits)))
                        e["followups"]+=1
                if need:
                    message=dict(rows=len(packet),data=base64.b64encode(b"".join(packet)).decode("ascii"))
                    if task.get("episodes"):
                        message["episodes"]=[offset+i for i,_ in need]
                    send(message)
                    reply=json.loads(sys.stdin.readline())
                    if len(reply)!=len(need):raise ValueError("wrong number of student moves")
                    for (i,index),a in zip(need,reply):actions[i]=index[a]
                for i in sorted(actions):
                    env=games[i];e=entries[i];tr=env._native.step(actions[i]);e["steps"]+=1
                    t=env.snapshot()
                    f=cohort[i][3]
                    ended=tr.exited_combat or tr.terminated or t["floor"]!=f["floor"] or t["decision"]["encounter"]!=f["encounter"]
                    if ended or e["steps"]>=400:
                        e["capped"]=not ended
                        e["hp"]=int(t["hp"]) if not e["capped"] else 0
                        e["win"]=bool(e["hp"]>0 and not e["capped"])
                        active.remove(i)
            summaries.extend(entries)
        send(dict(fights=summaries))


if __name__=="__main__":
    main()
