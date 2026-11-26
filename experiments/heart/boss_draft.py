"""Learn card reward values from matched boss practice and full-run continuations."""
import argparse
import hashlib
from pathlib import Path

import heart_native as engine
import numpy as np

from .deck_training import funnel
from .heart_curriculum import read_model, save_model
from .recovery import panel
from .replay import apply_model
from .train import write


def collect(parent,start,ordinary,late_count,draws,threads):
    weights=apply_model("first",parent)
    scout=np.asarray(engine.batch(weights,start+10000,32768,threads))
    late=start+10000+np.flatnonzero(scout[:,1]>=3)[:late_count]
    data=[]
    for act,seeds in ((1,np.arange(start,start+ordinary)),(3,late)):
        for offset in range(0,len(seeds),80):
            group=seeds[offset:offset+80]
            rows=np.asarray(engine.boss_draft_practice(weights,group.tolist(),act,draws,threads),np.float32)
            if rows.size:
                rows[:,0]=group[rows[:,0].astype(int)]-start
                data.append(rows)
    return np.concatenate(data)


def fit(rows,shrinkage):
    width=rows.shape[1]-6
    table=np.zeros((engine.card_count,width,2))
    for card in np.unique(rows[:,2]).astype(int):
        chosen=rows[rows[:,2]==card]
        x=chosen[:,4:-2].astype(np.float64)
        # The boss and the cards already held modify the value of this particular offer.
        # Pull sparse interactions toward zero to favor a per-card average on small samples.
        penalty=np.full(width,60.)
        penalty[0]=4.
        penalty[1:17]=30.
        penalty[17:26]=15.
        penalty[26:32]=40.
        active=x.any(axis=0)
        x=x[:,active]
        table[card,active]=np.linalg.solve(x.T@x+np.diag(penalty[active]*shrinkage),x.T@chosen[:,-2:])
    return table


def heldout(weights,rows):
    y=rows[:,-2:]
    predicted=np.einsum("ij,ijk->ik",rows[:,4:-2],weights[rows[:,2].astype(int)])
    return dict(rmse=np.sqrt(np.mean((y-predicted)**2,axis=0)).tolist(),
                zero_rmse=np.sqrt(np.mean(y**2,axis=0)).tolist(),
                correlations=[float(np.corrcoef(y[:,k],predicted[:,k])[0,1]) for k in range(2)])


def evaluate(parent,table,scales,start,count,threads):
    model={**parent,"boss_draft":(table@np.array(scales)).tolist()} if any(scales) else parent
    stats,rows=panel(model,start,count,threads)
    return model,{**stats,**funnel(rows)},rows


def rank(result):
    return (len(result["heart_encounter_seeds"]),result["act4"],result["second_boss"],
            result["first_boss"],result["act3"])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=Path("build/boss-draft"))
    parser.add_argument("--threads",type=int,default=10)
    parser.add_argument("--reuse",action="store_true")
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    parent=read_model("build/trajectory-teacher/challenger.json.gz")
    data={}
    report=dict(boss_draws=4,data={},validation=[],selection=[],test={})
    for name,start,normal,late in (("train",143000000,4096,640),("valid",144000000,768,128)):
        path=args.out/(name+".npz")
        if args.reuse:
            with np.load(path) as saved:
                rows=saved["rows"]
        else:
            rows=collect(parent,start,normal,late,4,args.threads)
            np.savez_compressed(path,rows=rows)
        data[name]=rows
        report["data"][name]=dict(start=start,ordinary=normal,late=late,rows=len(rows),
            comparisons_by_act=np.bincount(rows[:,3].astype(int),minlength=4)[1:].tolist(),
            decisions=len(np.unique(rows[:,:2],axis=0)))
        print(name,report["data"][name],flush=True)
    fits=[]
    for regularization in (.5,2.,8.):
        table=fit(data["train"],regularization)
        metric=heldout(table,data["valid"])
        fits.append((regularization,table,metric))
    reg,table,_=min(fits,key=lambda f:sum(np.array(f[2]["rmse"])/f[2]["zero_rmse"]))
    report["fits"]=[dict(regularization=r,**v) for r,_,v in fits]
    report["regularization"]=reg
    trials=[]
    for scales in ((12,2),(36,5),(80,12),(160,32),(320,64),(16,40),(180,3),(50,20)):
        _,stats,_=evaluate(parent,table,scales,145000000,3072,args.threads)
        record={**stats,"scales":list(scales)}
        trials.append((scales,stats))
        report["validation"].append(record)
    candidates=[]
    for scales,_ in sorted(trials,key=lambda tr:rank(tr[1]),reverse=True)[:3]:
        _,s,_=evaluate(parent,table,scales,146000000,8192,args.threads)
        candidates.append((scales,s))
        report["selection"].append({**s,"scales":list(scales)})
    scales,_=max(candidates,key=lambda x:rank(x[1]))
    report["selected"]=list(scales)
    print("selected",scales,report["fits"],flush=True)
    for start in (147000000,148000000):
        result={}
        for name,mix in (("parent",(0,0)),("combined",scales)):
            model,stats,rows=evaluate(parent,table,mix,start,32768,args.threads)
            result[name]=stats
            np.save(args.out/f"{name}-{start}.npy",rows)
            if name=="combined" and start==147000000:
                save_model(args.out/"challenger.json.gz",model)
            print(start,name,stats,flush=True)
        report["test"][start]=result
        write(args.out/"report.json",report)
    # Both signals can prescribe different cards. Isolate them on the same starts.
    report["components"]={}
    for name,mix in (("boss_only",(scales[0],0)),("run_only",(0,scales[1]))):
        _,stats,_=evaluate(parent,table,mix,147000000,32768,args.threads)
        report["components"][name]=stats
        print(name,stats,flush=True)
    report["checkpoint_sha256"]=hashlib.sha256((args.out/"challenger.json.gz").read_bytes()).hexdigest()
    write(args.out/"report.json",report)


if __name__=="__main__":
    main()
