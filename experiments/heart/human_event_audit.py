"""Check fitted event logits through the real action-scoring interface."""
from pathlib import Path

import heart_native as engine
import numpy as np
import sts_env as sts

from .expert_draft import enum
from .heart_curriculum import read_model
from .replay import apply_model
from .train import write


def main():
    event_screen=enum("build/rl-fork/include/game/GameContext.h","ScreenState").index("EVENT_SCREEN")
    out=Path("build/human-events")
    model=read_model(out/"context.json.gz")
    w=apply_model("first",model)
    table,seen,_=model["human_events"];table=np.asarray(table)
    zeros=np.zeros_like(table).tolist();params=table.tolist()
    menus=0;worst=0.;outside=0
    for seed in range(258900000,258900080):
        e=sts.Env();e.reset(seed,20,final_act=True,character="ironclad")
        while not e.done:
            o,mask=e.observe(),e.mask()
            q=np.asarray(engine.scores(w,o,mask))
            event=int(o[9])-1
            idx=np.flatnonzero(mask[sts.ACT_CHOOSE:sts.ACT_CHOOSE+4])
            if not o[7] and int(o[6])==event_screen and len(idx)>1 and 0<=event<len(seen) and seen[event]:
                engine.set_human_events(zeros,seen,1.)
                control=np.asarray(engine.scores(w,o,mask))
                engine.set_human_events(params,seen,1.)
                miss=1-o[0]/max(1,o[1]);local=max(0,int(o[3])-1)%17
                x=[1.,miss,miss**2,(o[1]-75)/50,min(o[2]/200,3.),local/15.,float(o[4]>=3)]
                actual=(q-control)[sts.ACT_CHOOSE+idx]
                expected=8*(table[event,idx]@x)
                worst=max(worst,float(np.max(np.abs(actual-expected))));menus+=1
            elif not o[7] and int(o[6])==event_screen and event>=0 and not seen[event] and len(idx)>1:
                engine.reset_human_events();base=np.asarray(engine.scores(w,o,mask))
                engine.set_human_events(params,seen,1.)
                if not np.array_equal(base,q):raise ValueError("unsupported event changed")
                outside+=1
            if not e.step(int(q.argmax())):raise ValueError("illegal move")
        if menus>=60 and outside:break
    if menus<20 or not outside or worst>1e-4:raise ValueError((menus,outside,worst))
    write(out/"audit.json",dict(menus=menus,unmapped_unchanged=outside,max_error=worst))


if __name__=="__main__":
    main()
