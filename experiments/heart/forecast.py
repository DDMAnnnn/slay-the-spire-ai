"""Learn visible enemy intent transitions and use them to plan across three future turns."""
import argparse
from pathlib import Path

import heart_native as engine
import numpy as np

from .heart_curriculum import read_model, save_model
from .replay import heart_encounters, load_model
from .train import statistics, write


def observations(w,start,count,threads):
    return np.asarray(engine.intent_practice(w,start,count,threads)).reshape(-1,18,3,4)


def fit(stats):
    n,total = stats[...,0],stats[...,1]
    mean = total.sum(axis=(0,1))/n.sum(axis=(0,1))
    enemy = (total.sum(axis=1)+20*mean)/(n.sum(axis=1)+20)
    return (total+20*enemy[:,None,:])/(n+20),mean


def accuracy(stats,table,mean):
    n,total,squares,persistence = np.moveaxis(stats,-1,0)
    sizes = n.sum(axis=(0,1))
    def rmse(errors):
        return np.sqrt(errors.sum(axis=(0,1))/sizes).round(3).tolist()
    def errors(pred):
        return squares-2*pred*total+n*pred**2
    return dict(pairs=sizes.astype(int).tolist(),carry_current_rmse=rmse(persistence),
                global_mean_rmse=rmse(errors(mean)),learned_rmse=rmse(errors(table)))


def mean_control(model,mean,start,threads):
    from .replay import apply_model
    params=model["forecast"]
    mean_table=np.broadcast_to(mean,np.asarray(params["intents"]).shape).tolist()
    w=apply_model("first",{**model,"forecast":{**params,"intents":mean_table}})
    rows=np.asarray(engine.batch(w,start,32768,threads))
    return dict(**statistics(rows),heart_encounter_seeds=heart_encounters(w,start,rows))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=Path("build/forecast"))
    parser.add_argument("--threads",type=int,default=10)
    args=parser.parse_args()
    args.out.mkdir(exist_ok=True,parents=True)
    ref=Path("build/first-boss-relics/challenger.json.gz")
    w=load_model("first",ref)
    training=observations(w,97000000,8192,args.threads)
    table,mean=fit(training)
    validation=accuracy(observations(w,98000000,2048,args.threads),table,mean)
    print("held-out forecast error",validation,flush=True)
    encounters=table.tolist()
    pool=np.asarray(engine.batch(w,99000000,12288,args.threads))
    banks=[engine.Bank(w,(99000000+np.flatnonzero(pool[:,0]>=floor)[:size]).tolist(),floor,args.threads)
           for floor,size in ((18,256),(35,180),(49,48))]
    rng=np.random.default_rng(2597)
    settings=[[float(rng.uniform(.4,3.0)),float(rng.uniform(.6,2.5)),
               float(rng.uniform(.1,1.2)),float(rng.uniform(.6,1.))] for _ in range(24)]
    proposals=[]
    for setting in settings:
        engine.set_forecast(encounters,setting)
        full=np.asarray(engine.batch(w,100000000,384,args.threads))
        s2,s3,boss=(np.asarray(b.progress(w,args.threads)) for b in banks)
        conditional=[float((full[:,1]>=2).mean()),float((s2[:,1]>=3).mean()),
                     float((s3[:,0]>=49).mean()),float((boss[:,1]==4).mean()),
                     float(boss[:,2].mean())]
        rank=sum(np.log(.035+np.asarray(conditional)))+.025*full[:,0].mean()
        proposals.append(dict(settings=setting,rank=float(rank),rates=conditional))
    top=sorted(proposals,key=lambda r:r["rank"],reverse=True)[:3]
    for candidate in top:
        engine.set_forecast(encounters,candidate["settings"])
        rows=np.asarray(engine.batch(w,101000000,4096,args.threads))
        candidate["validation"] = dict(**statistics(rows),encounters=len(heart_encounters(w,101000000,rows)))
    def rank(r):
        v=r["validation"]
        return v["encounters"],v["act4"],v["act3"],v["mean_floor"]
    best=max(top,key=rank)["settings"]
    model=read_model(ref)
    model["forecast"]=dict(intents=encounters,settings=best)
    challenger=args.out/"challenger.json.gz"
    save_model(challenger,model)
    print("full-run validation",[(c["validation"]["act3"],c["validation"]["encounters"]) for c in top],flush=True)
    start=102000000
    report=dict(intent_training_start=97000000,intent_training_runs=8192,
                training_pairs=training[...,0].sum(axis=(0,1)).astype(int).tolist(),
                forecast_validation_start=98000000,forecast_validation=validation,
                bank_sizes=[len(b) for b in banks],selection_seed=101000000,
                proposals=proposals,finalists=top,settings=best,test_start=start,test={})
    panels={}
    for name,path in (("reference",ref),("cross_turn",challenger)):
        weights=load_model("first",path)
        panel=np.asarray(engine.batch(weights,start,32768,args.threads))
        panels[name]=panel
        report["test"][name]=dict(**statistics(panel),heart_encounter_seeds=heart_encounters(weights,start,panel))
        print(name,report["test"][name],flush=True)
    a,b=panels.values()
    report["paired_act3"]=dict(gained=int(((a[:,1]<3)&(b[:,1]>=3)).sum()),
                               lost=int(((a[:,1]>=3)&(b[:,1]<3)).sum()))
    report["test"]["mean_attack_control"]=mean_control(model,mean,start,args.threads)
    write(args.out/"report.json",report)


if __name__=="__main__":
    main()
