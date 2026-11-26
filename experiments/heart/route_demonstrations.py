"""Export visible branching maps and the expert's route from terminal-hash-verified native games."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from public_demo_export import digest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--out",type=Path,default=Path("build/route-irl"))
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    root=Path("build/public-expert-traces")
    sys.path[:0]=[str((root/"agent"/k).resolve()) for k in ("native","runtime")]
    import slaythespire as sts
    from sts_rl_env import SpireEnv
    rows=games=0
    with gzip.open(root/"replays.jsonl.gz","rt") as src,gzip.open(args.out/"routes.tmp.gz","wt") as dst:
        for r in map(json.loads,src):
            if r["error"] or not r["public_information"] or not r["original_audit"]["passed"]:continue
            seed=r["seed"];env=SpireEnv(sts,sts.CharacterClass.IRONCLAD,seed,20,"snapshot")
            if digest(env.snapshot())!=r["start_state_hash"]:raise ValueError("invalid replay start")
            bucket=int.from_bytes(hashlib.sha256(str(seed).encode()).digest()[:4],"big")%10
            game=[];boards={}
            for domain,bits in r["trace"]:
                state=env.snapshot()
                if domain==0 and state["screen_state"]==5 and state["act"] in (1,2,3) and state["map_position"][1]<14 and bits in range(7):
                    act=state["act"];y=state["map_position"][1]
                    if act not in boards:
                        # Empty nodes and their nominal room types should never enter the decision score.
                        board=[[8,0] for _ in range(105)]
                        for x,yy,room,edges in state["map"]:
                            if 0<=yy<15 and 0<=x<7:
                                mask=sum(1<<(col-x+1) for col in edges if 0<=col<7 and abs(col-x)<=1)
                                board[7*yy+x]=[room,mask]
                        boards[act]=board
                    legal=sorted(a.bits for a in env._native.legal_actions() if a.domain==sts.ActionDomain(domain) and a.bits in range(7))
                    if bits not in legal:raise ValueError("route is absent from the visible map")
                    position=env._native.public_burning_elite_position()
                    if position is None:position=(-1,-1)
                    ctx=[act,state["hp"],state["max_hp"],state["gold"],sum(p>1 for p in state["potions"]),len(state["deck"]),*state["keys"]]
                    game.append([act,y+1,ctx,list(position),legal,bits])
                env._native.step(sts.EnvironmentAction(sts.ActionDomain(domain),bits))
            if digest(env.snapshot())!=r["terminal_state_hash"]:raise ValueError("invalid replay end")
            dst.write(json.dumps([seed,bucket,boards,game],separators=(",",":"))+"\n")
            rows+=len(game);games+=1
    (args.out/"routes.tmp.gz").replace(args.out/"routes.jsonl.gz")
    print("verified games",games,"route choices",rows,flush=True)


if __name__=="__main__":
    main()
