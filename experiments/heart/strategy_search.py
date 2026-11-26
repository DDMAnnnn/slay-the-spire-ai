"""Search combat, card rewards, resource use and routing as one full-run policy."""
import argparse
import math
from pathlib import Path

import heart_native as engine
import numpy as np

from .heart_curriculum import read_model, save_model
from .replay import heart_encounters, load_model, set_actor, set_combat_preferences, set_draft, set_late_combat, set_opening
from .train import NAMES, SCALE, statistics, write


class Space:
    def __init__(self):
        self.ref=read_model("build/first-boss-relics/challenger.json.gz")
        self.early=np.array(self.ref["weights"])
        self.late=np.array(self.ref["late"]["weights"])
        self.drafts=[None]+[read_model(Path("build/context-draft")/file)["draft_context"]
                            for file in ("id-only.json.gz","challenger.json.gz")]
        self.late_head=read_model("build/late-combat-hand/challenger.json.gz")["late_combat"]
        self.direct=read_model("build/direct-actor/second_round.json.gz")["direct_actor"]
        self.prefs=read_model("build/combat-preferences/challenger.json.gz")["combat_preferences"]
        self.opening=read_model("build/learned-opening/challenger.json.gz")["neow"]
        self.signed=[NAMES.index(n) for n in ("elite","rest","shop","hall","draft","swap","keyroute")]

    def build(self,offset,modes):
        synergy,draft,combat,drinks,bases,depth,opening=modes
        def weights(prior,direction):
            raw=prior+SCALE*direction
            bounded=raw.clip(.02)
            bounded[self.signed]=raw[self.signed]
            return bounded.tolist()
        early=weights(self.early,offset[0])
        late=weights(self.late,.65*offset[0]+.75*offset[1])
        pref={}
        if drinks:
            pref.update(drink_table=self.prefs["drink_table"],margin=self.prefs["margin"])
        if bases:
            pref["play_base"]=self.prefs["play_base"]
        return dict(weights=early,late=dict(act=2,weights=late),policy_synergies=bool(synergy),
                    draft_context=self.drafts[draft],late_combat=self.late_head if combat==1 else None,
                    late_combat_last_act=2,direct_actor=self.direct if combat==2 else None,
                    turn_depth=(2,3,4)[depth],neow=self.opening if opening else None,combat_preferences=pref)

    def apply(self,config):
        engine.set_late(2,config["late"]["weights"])
        engine.set_synergies(config["policy_synergies"])
        engine.set_turn_depth(config["turn_depth"])
        set_actor(config["direct_actor"])
        set_draft(config["draft_context"])
        set_opening(config["neow"])
        set_combat_preferences(config["combat_preferences"])
        set_late_combat(config["late_combat"])
        engine.set_late_combat_last_act(2)
        return config["weights"]


def recruit(start,n,threads):
    w=load_model("first","build/heart-curriculum/staged.json.gz")
    rows=np.array(engine.batch(w,start,n,threads))
    return [engine.Bank(w,(start+np.flatnonzero(rows[:,0]>=floor)[:limit]).tolist(),floor,threads)
            for floor,limit in ((18,192),(35,192),(49,128))]


def score(w,banks,start,n,threads):
    full=np.array(engine.batch(w,start,n,threads))
    act2,act3,boss=(np.asarray(b.progress(w,threads)) for b in banks)
    def log_rate(x):
        return math.log((2.+x.sum())/(4.+len(x)))
    # Conditional survival estimates keep later decisions visible when a small
    # ordinary-start batch contains no Heart fight. Full starts remain the final test.
    return (log_rate(full[:,1]>=2)+log_rate(act2[:,1]>=3)+log_rate(act3[:,0]>=49)
            +log_rate(.7*boss[:,2]+.2*(boss[:,1]==4)+.1*(boss[:,0]>=51))
            +.02*full[:,0].mean())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=Path("build/joint-strategy"))
    parser.add_argument("--threads",type=int,default=10)
    args=parser.parse_args()
    args.out.mkdir(exist_ok=True,parents=True)
    train=recruit(88000000,16384,args.threads)
    valid=recruit(89000000,16384,args.threads)
    load_model("first","build/first-boss-relics/challenger.json.gz")
    space=Space()
    rng=np.random.default_rng(880956)
    sizes=[2,3,3,2,2,3,2]
    centre=np.zeros((2,len(NAMES)))
    spread=np.full(centre.shape,3.5)
    rates=[np.ones(n)/n for n in sizes]
    finalists=[]
    for gen in range(8):
        population=[]
        for j in range(20):
            offset=centre+spread*rng.standard_normal(centre.shape)
            modes=[int(rng.choice(n,p=p)) for n,p in zip(sizes,rates)]
            config=space.build(offset,modes)
            value=score(space.apply(config),train,90000000+512*gen,160,args.threads)
            population.append((value,offset,modes))
        elite=sorted(population,key=lambda t:t[0],reverse=True)[:5]
        moves=np.stack([row[1] for row in elite])
        centre=.25*centre+.75*moves.mean(axis=0)
        spread=np.maximum(.8,.4*spread+.6*moves.std(axis=0))
        for i,n in enumerate(sizes):
            counts=np.bincount([row[2][i] for row in elite],minlength=n)+.45
            rates[i]=.3*rates[i]+.7*counts/counts.sum()
        value,offset,modes=elite[0]
        rank=score(space.apply(space.build(offset,modes)),valid,91000000,512,args.threads)
        finalists.append((rank,offset,modes))
        print("generation",gen+1,"train",round(value,3),"conditional validation",round(rank,3),flush=True)
    finalists=sorted(finalists,key=lambda t:t[0],reverse=True)[:3]
    panel=[]
    for fit,offset,modes in finalists:
        config=space.build(offset,modes)
        w=space.apply(config)
        rows=np.asarray(engine.batch(w,92000000,4096,args.threads))
        encounters=heart_encounters(w,92000000,rows)
        # Prefer observed endgame progress. Use the conditional estimate for ties.
        key=(len(encounters),int((rows[:,1]==4).sum()),fit)
        panel.append((key,config,modes,statistics(rows)))
    key,champion,modes,_=max(panel,key=lambda t:t[0])
    model={**space.ref,**champion}
    challenger=args.out/"challenger.json.gz"
    save_model(challenger,model)
    report=dict(training_banks=[len(b) for b in train],validation_banks=[len(b) for b in valid],
                candidates=160,selected_modes=modes,weights=champion["weights"],late_weights=champion["late"]["weights"],
                validation=[dict(rank=k,structural_choices=m,**stats) for k,_,m,stats in panel],
                test_start=93000000,test={})
    for name,path in (("reference",Path("build/first-boss-relics/challenger.json.gz")),("joint",challenger)):
        w=load_model("first",path)
        result=np.array(engine.batch(w,93000000,32768,args.threads))
        summary=dict(**statistics(result),heart_encounter_seeds=heart_encounters(w,93000000,result))
        print(name,summary,flush=True)
        report["test"][name]=summary
        write(args.out/"report.json",report)


if __name__=="__main__":
    main()
