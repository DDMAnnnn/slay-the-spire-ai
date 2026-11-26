"""Export source strategic choices only after replay verifies the entire recorded game."""
from array import array
import base64
import gzip
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from public_demo_export import digest
from strategic_projection import StrategicProjection


def main():
    root=Path("build/public-expert-traces")
    out=Path("build/strategic-apprentice");out.mkdir(exist_ok=True,parents=True)
    sys.path[:0]=[str((root/"agent"/k).resolve()) for k in ("native","runtime")]
    import slaythespire as sts
    from sts_rl_env import SpireEnv
    codec=StrategicProjection();games=rows=0
    dest=out/"choices.jsonl.gz";tmp=dest.with_suffix(".tmp")
    with gzip.open(root/"replays.jsonl.gz","rt") as src,gzip.open(tmp,"wt",compresslevel=3) as fp:
        for r in map(json.loads,src):
            if r["error"] or not r["public_information"] or not r["original_audit"]["passed"]:continue
            env=SpireEnv(sts,sts.CharacterClass.IRONCLAD,r["seed"],20,"snapshot")
            if digest(env.snapshot())!=r["start_state_hash"]:raise ValueError("wrong source reset")
            game=[]
            for domain,bits in r["trace"]:
                if domain==0 and not bits&0x80000000:
                    s=env.snapshot()
                    if s["screen_state"]!=5 and not (s["screen_state"]==1 and s["current_event"]==31):
                        legal=[a.bits for a in env._native.legal_actions()]
                        if sum(not a&0x80000000 for a in legal)>1:
                            o,m=codec.encode(s,legal,env._native.public_burning_elite_position())
                            picked=next((a for a,b in m.items() if b==bits),None)
                            if picked is not None:
                                # Integers suffice for these public noncombat fields; the final boss edge's raw bit is unused.
                                packet=base64.b64encode(array("h",[int(v) for v in o]).tobytes()).decode("ascii")
                                game.append([packet,list(m),picked])
                env._native.step(sts.EnvironmentAction(sts.ActionDomain(domain),bits))
            if digest(env.snapshot())!=r["terminal_state_hash"]:raise ValueError("diverged before terminal")
            for row in game:fp.write(json.dumps([r["seed"],*row],separators=(",",":"))+"\n")
            rows+=len(game);games+=1
    tmp.replace(dest);print(games,rows,flush=True)


if __name__=="__main__":main()
