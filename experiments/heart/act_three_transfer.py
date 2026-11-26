"""Test Act 3 combat transfer on ordinary starts, keeping the earned prefix unchanged."""
import json
from pathlib import Path

import heart_native as engine
import numpy as np
import torch

from .heart_curriculum import read_model
from .late_demo_policy import play, summary
from .public_demo_student import Actor
from .replay import apply_model
from .train import write
from .transfer_probe import EXAMPLE, model_code


def main():
    out=Path("build/act-three-transfer");out.mkdir(parents=True,exist_ok=True)
    file=out/"report.json";r=json.loads(file.read_text()) if file.exists() else {}
    torch.set_num_threads(2);device="cuda" if torch.cuda.is_available() else "cpu"
    parent=read_model("build/stratified-gate/challenger.json.gz");w=apply_model("first",parent)
    start=246000000;ids=np.arange(start,start+32768)
    path=out/"control.npy"
    if not path.exists():np.save(path,np.asarray(engine.batch_seeds(w,ids.tolist(),3)))
    control=np.load(path)
    # The source experiment showed its largest held-out gain specifically in Act 3.
    ix=np.flatnonzero(control[:,0]>=35)
    ref=control[:,[0,1,6,4,5]]
    r.update(start=start,eligible=len(ix),parent=summary(ref),parent_hits=ids[ref[:,2]>0].tolist())
    net,_=model_code().load(EXAMPLE/"weights.pt",device)
    actor=Actor(net.eval(),device,w,3,True);actor.blend=.65
    if "prefix_control" not in r:
        z=play(ids[ix[:32]],99,actor,device,3)
        if not np.array_equal(z,ref[ix[:32]]):raise ValueError("native prefix control differs")
        r["prefix_control"]=len(z);write(file,r)
    for name,checkpoint in (("imitation","build/public-expert-intent/imitation.pt"),
                            ("reinforcement","build/matched-combat-rl/selected.pt")):
        if name in r:continue
        net.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True));net.eval()
        result=ref.copy()
        for i in range(0,len(ix),128):
            part=ix[i:i+128];path=out/f"{name}-{i}.npy"
            if not path.exists():np.save(path,play(ids[part],35,actor,device,3,max_act=3))
            result[part]=np.load(path)
        hits=result[:,2]>0;old=ref[:,2]>0
        r[name]=dict(**summary(result),heart_encounter_seeds=ids[hits].tolist(),victory_seeds=ids[result[:,3]>0].tolist(),
                     gained=int((hits&~old).sum()),lost=int((old&~hits).sum()))
        write(file,r);print(name,r[name],flush=True)


if __name__=="__main__":
    main()
