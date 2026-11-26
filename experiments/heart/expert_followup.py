"""Confirm the boss-relic component on fresh panels alongside the simulated card prior."""
from pathlib import Path

import numpy as np

from .deck_training import funnel
from .heart_curriculum import read_model,save_model
from .recovery import panel
from .train import write


def main():
    parent=read_model("build/trajectory-teacher/challenger.json.gz")
    boss=read_model("build/boss-draft/challenger.json.gz")
    expert=read_model("build/expert-draft/challenger.json.gz")
    cards,relics,neow=expert["expert_prior"]
    only=[np.zeros_like(cards).tolist(),relics,np.zeros_like(neow).tolist()]
    models={"parent":parent,"boss_relics":{**parent,"expert_prior":only},
            "combined":{**boss,"expert_prior":only},"boss_draft":boss}
    out=Path("build/expert-followup")
    out.mkdir(exist_ok=True,parents=True)
    result={}
    for start in (153000000,154000000):
        row={}
        for name,model in models.items():
            stats,r=panel(model,start,32768,10)
            row[name]={**stats,**funnel(r)}
            print(start,name,row[name],flush=True)
            if start==153000000 and name in ("boss_relics","combined"):
                save_model(out/(name+".json.gz"),model)
        result[start]=row
        write(out/"report.json",result)


if __name__=="__main__":
    main()
