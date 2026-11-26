"""Check cross-runtime strategic action agreement on public positions shared by both simulators."""
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import subprocess
import sys


def source_export(destination, seeds):
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from strategic_projection import StrategicProjection
    root=Path("build/public-expert-traces");sys.path[:0]=[str((root/"agent"/k).resolve()) for k in ("native","runtime")]
    import slaythespire as sts
    from sts_rl_env import SpireEnv
    codec=StrategicProjection()
    with gzip.open(root/"replays.jsonl.gz","rt") as src,gzip.open(destination,"wt") as dst:
        for r in map(json.loads,src):
            if r["seed"] not in seeds:continue
            env=SpireEnv(sts,sts.CharacterClass.IRONCLAD,r["seed"],20,"snapshot")
            for domain,bits in r["trace"]:
                if domain==0:
                    s=env.snapshot()
                    if not (s["screen_state"]==1 and s["current_event"]==31):
                        legal=[a.bits for a in env._native.legal_actions()]
                        if len(legal)>1:
                            o,m=codec.encode(s,legal,env._native.public_burning_elite_position())
                            sparse=[(i,v) for i,v in enumerate(o) if v]
                            dst.write(json.dumps([r["seed"],sparse,list(m)],separators=(",",":"))+"\n")
                env._native.step(sts.EnvironmentAction(sts.ActionDomain(domain),bits))


def main():
    import heart_native as engine
    import numpy as np
    import sts_env as sts
    from .heart_curriculum import read_model
    from .replay import apply_model, rows
    from .train import write
    out=Path("build/source-strategy");out.mkdir(exist_ok=True,parents=True)
    with gzip.open("build/public-expert-traces/converted.jsonl.gz","rt") as src:
        games=sorted(map(json.loads,src),key=lambda r:len(r["actions"]),reverse=True)[:96]
    file=out/"paired-projections.jsonl.gz"
    if not file.exists():
        tmp=file.with_suffix(".tmp")
        cmd=[str(Path("build/public-expert-traces/python313/python.exe").resolve()),str(Path(__file__)),str(tmp)]
        subprocess.run(cmd,input=json.dumps([v["seed"] for v in games]),text=True,check=True)
        tmp.replace(file)
    w=apply_model("first",read_model("build/route-irl/selected-whole_map.json.gz"))
    def abst(o,mask):
        choice=rows(o,"choices")
        def label(a):
            if a<sts.ACT_CHOOSE:return ("potion",int(a))
            i=a-sts.ACT_CHOOSE
            return (int(choice[i,0]),*map(float,choice[i,1:4]))
        options={int(a):label(a) for a in np.flatnonzero(mask)}
        deck=tuple(sorted(tuple(map(int,c)) for c in rows(o,"deck") if c[0]))
        key=(tuple(o[[0,1,2,3,4,6,9,10,11]].astype(int)),deck,tuple(sorted(map(str,options.values()))))
        q=np.asarray(engine.scores(w,o,mask))
        return key, options[int(q.argmax())],{str(k):round(max(q[a] for a in options if options[a]==k),6) for k in options.values()}
    shared=defaultdict(dict)
    for g in games:
        env=sts.Env();env.reset(g["seed"],20,final_act=True,character="ironclad")
        for a in g["actions"]:
            o,m=env.observe(),env.mask()
            if not o[7] and m.sum()>1:
                key,chosen,scores=abst(o,m);shared[g["seed"]][key]=(chosen,scores)
            if not env.step(a):raise ValueError("stored local prefix failed to reproduce")
    by_position={seed:{k[:2]:k for k in opts} for seed,opts in shared.items()}
    counts=defaultdict(Counter);mismatch=Counter();examples={}
    with gzip.open(file,"rt") as src:
        for seed,fields,legal in map(json.loads,src):
            o=np.zeros(sts.OBS_SIZE,np.float32);m=np.zeros(sts.N_ACTIONS,np.uint8)
            for i,v in fields:o[i]=v
            m[legal]=1
            key,pick,scores=abst(o,m)
            rec=shared[seed].get(key)
            if rec is None:
                screen=int(o[6]);other=by_position[seed].get(key[:2])
                if other is not None and screen not in examples and key[2]!=other[2]:
                    examples[screen]=dict(source=sorted(set(key[2])-set(other[2])),local=sorted(set(other[2])-set(key[2])))
                continue
            del shared[seed][key]
            screen=int(o[6]);counts[screen]["shared"]+=1;counts[screen]["same_action"]+=pick==rec[0]
            err=max(abs(scores[k]-v) for k,v in rec[1].items())
            counts[screen]["same_scores"]+=int(err<.0001)
            if pick!=rec[0]:mismatch[(screen,str(pick),str(rec[0]))]+=1
    result={"by_screen":dict(counts),"mismatches":mismatch.most_common(15),"source_games":len(games),"menu_differences":examples}
    if sum(c["shared"] for c in counts.values())<100:raise ValueError("too few matched public decisions")
    write(out/"projection-audit.json",result);print(result,flush=True)


if __name__=="__main__":
    if len(sys.argv)>1:source_export(Path(sys.argv[1]),set(json.loads(input())))
    else:main()
