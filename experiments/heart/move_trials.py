"""Measure the effect of swapping two plausible moves on paired public combat futures."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from combat_afterstate import choose, competing_moves, pair_features
from public_demo_export import digest


def finish(world, first, sts, seed):
    native=world._native
    tr=native.step(sts.EnvironmentAction(sts.ActionDomain(1),first))
    rng=random.Random(seed)
    for _ in range(240):
        if tr.exited_combat or tr.terminated:
            s=world.snapshot()
            hp=max(0,s["hp"])
            return 35.+hp+4.*sum(v>1 for v in s["potions"]) if hp else 0.
        legal=native.legal_actions()
        if len(legal)==1:
            action=legal[0]
        elif any(a.bits==0x80000000 for a in legal):
            action=native.public_belief_combat_search_action(192,4,rng.getrandbits(53),2,True).action
        else:
            action=min(legal,key=lambda a:int(a.bits))
        tr=native.step(action)
    return None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard",type=int,required=True)
    parser.add_argument("--shards",type=int,default=4)
    parser.add_argument("--limit",type=int,default=400)
    args=parser.parse_args()
    root=Path("build/public-expert-traces")
    sys.path[:0]=[str((root/"agent"/k).resolve()) for k in ("native","runtime")]
    import slaythespire as sts
    from sts_rl_env import SpireEnv
    folder=Path("build/move-trials");folder.mkdir(parents=True,exist_ok=True)
    model=json.loads(Path("build/afterstate-critic/critic.json").read_text())
    layout=json.loads(Path("build/public-expert-intent/layout.json").read_text())
    layout.update(critic_cards=model["cards"],critic_enemies=model["enemies"])
    with gzip.open(root/"replays.jsonl.gz","rt",encoding="utf8") as fp:
        for number,row in enumerate(map(json.loads,fp)):
            if number>=args.limit:break
            if number%args.shards!=args.shard or row["error"] or not row["public_information"] or not row["original_audit"]["passed"]:continue
            path=folder/f"game-{number}.json"
            if path.exists():continue
            rng=random.Random(280771+number)
            fights=row["fights"]
            if not fights:continue
            # Draw whole fights before replay so late acts can enter the sample.
            chosen=sorted(range(len(fights)),key=lambda i:rng.random()**(1/(1+int(fights[i]["act"]>=2))),reverse=True)[:4]
            goals={i:rng.randrange(9) for i in chosen}
            examples=[];prev=None;fight=-1
            env=SpireEnv(sts,sts.CharacterClass.IRONCLAD,row["seed"],20,"snapshot")
            for d,b in row["trace"]:
                state=env.snapshot()
                current=(state["act"],state["floor"],state["decision"]["encounter"]) if state["in_combat"] else None
                if current and current!=prev:fight+=1
                prev=current
                if fight in goals and d==1:
                    legal=env._native.legal_actions()
                    if len(legal)>1 and any(a.bits==0x80000000 for a in legal):
                        goals[fight]-=1
                        if goals[fight]<0:
                            seed=rng.getrandbits(53)
                            base=env._native.public_belief_combat_search_action(384,4,seed,2,True)
                            diagnostic={}
                            _,compared=choose(env,sts,state,base,seed,model,layout,1.,3,diagnostic)
                            if compared:
                                left=int(base.action.bits);_,right=competing_moves(base)
                                x=pair_features(state,base,diagnostic,layout)
                                margins=[]
                                for trial in range(3):
                                    # Independent of the critic's three feature worlds.
                                    chance=rng.getrandbits(53)
                                    world=env.clone_with_resampled_hidden(chance)
                                    a=finish(world.clone(),left,sts,chance)
                                    bvalue=finish(world.clone(),right,sts,chance)
                                    if a is not None and bvalue is not None:margins.append(bvalue-a)
                                if len(margins)==3:
                                    examples.append(dict(fight=fight,x=x,delta=margins))
                                del goals[fight]
                env._native.step(sts.EnvironmentAction(sts.ActionDomain(d),b))
            if digest(env.snapshot())!=row["terminal_state_hash"]:
                raise ValueError(("counterfactuals altered the real run",row["seed"]))
            bucket=int.from_bytes(hashlib.sha256(str(row["seed"]).encode()).digest()[:4],"big")%10
            split="train" if bucket<8 else "validation" if bucket==8 else "test"
            tmp=path.with_suffix(".tmp");tmp.write_text(json.dumps(dict(seed=str(row["seed"]),split=split,rows=examples)));tmp.replace(path)


if __name__=="__main__":
    main()
