"""Verify the score interface on public hands, including invariance to draw-pile listing order."""
import argparse
from pathlib import Path

import numpy as np
import heart_native as engine
import sts_env as sts

from .heart_curriculum import read_model
from .replay import apply_model,rows,label
from .train import write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=Path("build/public-belief/audit.json"))
    args=p.parse_args()
    parent=read_model("build/trajectory-teacher/challenger.json.gz")
    settings=dict(seed=190000000,count=3072,draws=3,scope=1,scale=8.)
    model={**parent,"public_belief":settings}
    w=apply_model("first",model)
    drift=changed=checked=0
    example=None
    for seed in range(195000000,195000012):
        env=sts.Env()
        env.reset(seed,20,final_act=True,character="ironclad")
        for step in range(3500):
            if env.done:break
            obs=env.observe().copy();mask=env.mask().copy()
            scores=np.asarray(engine.scores(w,obs,mask))
            action=int(scores.argmax())
            if obs[7] and mask[sts.ACT_END_TURN] and mask.sum()>1:
                copy=obs.copy()
                pile=rows(copy,"draw");n=int((pile[:,0]>0).sum())
                if n>1:
                    pile[:n]=pile[:n][::-1].copy()
                    permuted=np.asarray(engine.scores(w,copy,mask))
                    drift=max(drift,float(np.max(abs(scores-permuted))))
                    checked+=1
                engine.set_belief(0,0,0.)
                prior=np.asarray(engine.scores(w,obs,mask))
                engine.set_belief(settings["draws"],settings["scope"],settings["scale"])
                if prior.argmax()!=action:
                    changed+=1
                    if example is None:
                        example=dict(seed=seed,step=step,floor=int(obs[3]),hp=int(obs[0]),
                                     before=label(obs,int(prior.argmax())),after=label(obs,action))
            if not mask[action] or not env.step(action):raise RuntimeError("illegal public action")
    if drift or not changed:raise RuntimeError(f"draw order drift {drift}, changed {changed}")
    coupled=engine.paired_draw_audit(256)
    if coupled[1]!=256 or not all(coupled[2:13]):
        raise AssertionError("opening permutation coupling failed")
    result=dict(root_order_checks=checked,max_score_change=drift,overrides=changed,example=example,
                original_order_preserved=dict(ordinary=int(coupled[0]),paired=int(coupled[1]),draws=256),
                inserted_card_position_counts=list(coupled[2:13]))
    write(args.out,result)
    print(result)


if __name__=="__main__":
    main()
