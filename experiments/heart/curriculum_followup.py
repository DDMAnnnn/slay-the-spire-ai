"""Separate combat and macro corrections and repeat the frozen comparison."""
from pathlib import Path

from .deck_training import funnel
from .heart_curriculum import read_model
from .recovery import panel
from .train import write


def main():
    out=Path("build/matched-curriculum")
    parent=read_model("build/trajectory-teacher/challenger.json.gz")
    model=read_model(out/"challenger.json.gz")
    report={"components":{},"confirmation":{}}
    for scope,name in ((1,"combat"),(2,"macro")):
        variant={**model,"curriculum_actor":[*model["curriculum_actor"],scope]}
        s,r=panel(variant,163000000,32768,10)
        report["components"][name]={**s,**funnel(r)}
        print(name,report["components"][name],flush=True)
    for name,variant in (("parent",parent),("curriculum",model)):
        s,r=panel(variant,165000000,32768,10)
        report["confirmation"][name]={**s,**funnel(r)}
        print(name,report["confirmation"][name],flush=True)
    write(out/"followup.json",report)


if __name__=="__main__":
    main()
