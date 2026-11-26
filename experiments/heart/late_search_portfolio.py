"""Change tactical search only after an ordinary run earns its first Act 3 boss."""
import json
from pathlib import Path

import heart_native as engine
import numpy as np

from .heart_curriculum import read_model
from .replay import apply_model
from .train import write


def main():
    out=Path("build/late-search-portfolio");out.mkdir(exist_ok=True,parents=True)
    source=Path("build/route-search-transfer/joint");base=read_model(source/"map.json.gz")
    raw=np.load(source/"194000000-map.npy");report=json.loads((source/"report.json").read_text())
    reached=set(report["map"]["heart_encounter_seeds"])
    expected=np.column_stack((raw[:,0],raw[:,1],[194000000+i in reached for i in range(len(raw))],raw[:,4],raw[:,5]))
    eligible=np.flatnonzero(raw[:,0]>=50)
    order=np.random.default_rng(276771).permutation(eligible)
    w=apply_model("first",base)
    engine.set_belief(0,1,8.);engine.enrich_late_particles(w,192)
    costs=Path("build/resource-effects")
    tables=json.loads((costs/"tables.json").read_text());typical=json.loads((costs/"fit.json").read_text())["typical_potion_cost"]
    options={"parent":(3,1,8.,1,32.,False,0.),"deep":(8,2,12.,3,90.,True,0.),
             "survive":(6,2,8.,1,90.,True,0.),"resources":(6,2,8.,1,90.,True,1.)}
    def enable(name):
        draws,scope,scale,plans,win,extra,mix=options[name]
        engine.set_belief(draws,scope,scale);engine.set_belief_plans(plans,win,extra)
        engine.set_stasis_reconstruction(extra)
        if mix:engine.set_resource_effects(tables["hp"],tables["potion"],0.,mix,typical)
        else:engine.reset_resource_effects()
    path=out/"report.json";r=json.loads(path.read_text()) if path.exists() else {"eligible":len(eligible),"ordinary_starts":len(raw)}
    def evaluate(group,ids,names):
        if group in r:return
        parts={k:[] for k in names}
        for off in range(0,len(ids),24):
            ix=ids[off:off+24];files={k:out/f"{group}-{k}-{off}.npy" for k in names}
            if not all(f.exists() for f in files.values()):
                enable("parent")
                bank=engine.Bank(w,(194000000+ix).tolist(),50,3)
                for k,f in files.items():
                    if not f.exists():
                        enable(k);arr=np.asarray(bank.progress(w,3));np.save(f,arr)
            for k,f in files.items():parts[k].append(np.load(f))
        control=np.concatenate(parts["parent"])
        if not np.array_equal(control,expected[ids]):raise ValueError("unchanged gated continuation did not reproduce")
        result={}
        for k,chunks in parts.items():
            a=np.concatenate(chunks);hit=a[:,2]>0;old=control[:,2]>0
            result[k]=dict(n=len(a),encounters=int(hit.sum()),kills=int(a[:,3].sum()),second=int((a[:,0]>=51).sum()),
                           act4=int((a[:,1]==4).sum()),gained=int((hit&~old).sum()),lost=int((~hit&old).sum()),capped=int(a[:,4].sum()))
        r[group]=result;write(path,r);print(group,result,flush=True)
    evaluate("selection",order[:144],list(options))
    def rank(k):
        v=r["selection"][k];return v["encounters"],v["kills"],v["act4"],v["second"]
    best=max((k for k in options if k!="parent"),key=rank);r["selected"]=best
    evaluate("test",order[144:],["parent",best]);write(path,r)


if __name__=="__main__":main()
