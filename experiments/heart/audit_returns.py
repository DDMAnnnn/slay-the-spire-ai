"""Check native score differences against exported weights on recorded public states."""
import argparse
from pathlib import Path

import heart_native as engine
import numpy as np
import sts_env as sts

from .heart_curriculum import read_model
from .replay import apply_model
from .train import write


def audit(model,data,seed,macro):
    base=read_model("build/trajectory-teacher/challenger.json.gz")
    weights=apply_model("first",base)
    with np.load(data) as z:
        rows=z["examples" if macro else "rows"]
    samples=[]
    for i in range(4):
        subset=rows[rows[:,0]==i]
        groups={int(g[0,1]):g for g in np.split(subset,1+np.flatnonzero(np.diff(subset[:,1]))) if len(g)}
        if not groups: continue
        game=sts.Env()
        game.reset(seed+i,20,final_act=True,character="ironclad")
        for step in range(max(groups)+1):
            obs=game.observe().copy();mask=game.mask().copy()
            prior=np.asarray(engine.scores(weights,obs,mask))
            chosen=int(prior.argmax())
            if step in groups:
                group=groups[step]
                ids=group[:,2].astype(int)
                if ids[0]!=chosen or not np.all(mask[ids]):
                    raise RuntimeError(f"offline prefix mismatch at seed {seed+i}, decision {step}")
                samples.append((obs,mask,prior,group))
                if macro:
                    k=max(range(len(group)),key=lambda j:tuple(group[j,4:9]))
                    chosen=ids[k]
            if not game.step(chosen): raise RuntimeError("invalid audit action")
    weights=apply_model("first",model)
    attention=model.get("choice_attention")
    if attention:
        tokens,foes,layer,bias,query,key,value,mix,out,scale=attention
        query,key,value,mix=map(np.asarray,(query,key,value,mix))
        scope=6
    else:
        tokens,foes,layer,bias,out,scale,scope=model["curriculum_actor"]
    tokens,foes,layer,bias,out=map(np.asarray,(tokens,foes,layer,bias,out))
    errors=[]
    for obs,mask,prior,g in samples:
        ids=g[:,2].astype(int)
        features=g[:,9:] if macro else g[:,11:]
        targets=np.zeros(len(g),int) if macro else g[:,4].astype(int)
        hidden=features@layer.T+bias+tokens[g[:,3].astype(int)]+foes[targets]
        linear=np.maximum(0,hidden)@out
        gap=(prior[ids]-prior[ids[0]])/6
        if attention:
            h=np.maximum(0,hidden)
            def score(h):
                logits=(h@query.T)@(h@key.T).T/8**.5
                weights=np.exp(logits-logits.max(1,keepdims=True))
                context=(weights/weights.sum(1,keepdims=True))@(h@value.T)
                return np.maximum(0,h+context@mix.T)@out
            linear=score(h)
            if np.max(abs(linear-score(h[::-1])[::-1]))>1e-7:
                raise RuntimeError("permuting the legal options changed the predicted utility")
            offset=8*np.tanh((linear-linear[0])/8)
        elif scope==4:
            offset=8*np.tanh((linear-linear[0])/8)
        else:
            offset=8*np.tanh(linear/8)
            if scope==5: offset-=offset[0]
        if scope==3: gap=np.maximum(gap,-10)
        elif not attention: offset[gap < -7]=0.
        expected=6*(gap+scale*offset)
        actual=np.asarray(engine.scores(weights,obs,mask))[ids]
        errors.extend(abs((expected-expected[0])-(actual-actual[0])))
    error=float(max(errors))
    if error>2e-3: raise RuntimeError(f"native score difference mismatch {error}")
    return dict(seed=seed,states=len(samples),candidate_scores=len(errors),max_error=error)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("model",type=Path)
    p.add_argument("data",type=Path)
    p.add_argument("--macro",action="store_true")
    args=p.parse_args()
    result=audit(read_model(args.model),args.data,int(args.data.stem),args.macro)
    print(result)
    write(args.model.parent/"parity.json",result)
