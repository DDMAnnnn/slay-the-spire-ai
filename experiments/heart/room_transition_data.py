"""Train a stochastic room/HP transition table and the value at the planning horizon."""
import json
from pathlib import Path

import heart_native as engine
import numpy as np

from .heart_curriculum import read_model
from .replay import apply_model
from .train import write

H=9
# terminal labels are probabilities of earning entry to the following act.


def normalize(counts,prior,weight):
    return (counts+weight*prior)/(counts.sum(-1,keepdims=True)+weight)


def main():
    out=Path("build/room-transitions");out.mkdir(exist_ok=True,parents=True)
    model=read_model("build/human-macro-replication/fire.json.gz");w=apply_model("first",model)
    ids=np.random.default_rng(270771).bit_generator.random_raw(32768)
    arrays=[]
    for start in range(0,len(ids),512):
        path=out/f"training-{start}.npy"
        if not path.exists():
            games=engine.room_transitions(w,ids[start:start+512].tolist(),4)
            parts=[]
            for j,rows in enumerate(games):
                if rows:
                    a=np.asarray(rows,dtype=np.int32)
                    if np.any(a[:,0]!=j):raise ValueError("wrong game index")
                    if a[:,20].any():continue
                    a[:,0]+=start;parts.append(a)
            np.save(path,np.concatenate(parts))
        arrays.append(np.load(path))
    a=np.concatenate(arrays)
    seed,act,row,hp,maxhp,potions,_,_,_,keys,room,burning,_,parent,_,after,maximum,_,alive,end_act,_=a.T
    act=act-1;stage=np.minimum(3,row//4);p=(potions>0).astype(int)
    green=(keys>>1)&1
    tile=np.where((room==3)&(burning==1),8,room)
    h=np.rint((H-1)*np.clip(hp/maxhp,0,1)).astype(int)
    target=np.where(alive>0,1+np.rint((H-1)*np.clip(after/maximum,0,1)).astype(int),0)
    bucket=(seed.astype(np.int64)*2654435761 % 2**32)%10
    train=bucket<8;validation=bucket==8;test=bucket==9
    # Smooth rare cells toward adjacent HP levels and then toward the coarser act/room table.
    def fit(shrink,adjacent):
        counts=np.zeros((3,4,2,9,H,H+1))
        np.add.at(counts,(act[train],stage[train],p[train],tile[train],h[train],target[train]),1)
        # Unobserved room groups stay close to an identity HP transition.
        identity=np.zeros((H,H+1));identity[np.arange(H),1+np.arange(H)]=1.
        pooled=counts.sum((1,2))
        coarse=normalize(pooled,identity,3.)
        neighbors=counts.copy()
        neighbors[...,1:,:]+=adjacent*counts[...,:-1,:]
        neighbors[...,:-1,:]+=adjacent*counts[...,1:,:]
        trans=normalize(neighbors,coarse[:,None,None],shrink)
        # Use units from one natural root per sampled decision, regardless of how many alternatives it offered.
        m=train & (parent==1)
        value_counts=np.zeros((3,4,2,2,H,2))
        advanced=(end_act>act+1).astype(int)
        np.add.at(value_counts,(act[m],stage[m],p[m],green[m],h[m],advanced[m]),1)
        global_rate=normalize(value_counts.sum((1,2,3,4)),np.array([.5,.5]),1.)
        by_hp=normalize(value_counts.sum((1,2,3)),global_rate[:,None,:],5.)
        value=normalize(value_counts,by_hp[:,None,None,None,:,:],shrink)[...,1]
        return trans,value,coarse
    def stats(trans,selection):
        dist=trans[act[selection],stage[selection],p[selection],tile[selection],h[selection]]
        y=target[selection]
        return dict(rooms=len(y),nll=float(-np.log(dist[np.arange(len(y)),y].clip(1.e-8)).mean()),
                    death_brier=float(np.mean((dist[:,0]-(y==0))**2)))
    fits={str((k,adj)):fit(k,adj) for k in (4.,16.,64.,256.) for adj in (0.,.25)}
    full,value,coarse=fits[str((64.,0.))]
    pooled=np.broadcast_to(coarse[:,None,None],full.shape)
    fits["pooled"]=(pooled,value,coarse)
    # Sparse cells can teach room mortality more reliably than the entire exit-HP distribution.
    conditional=pooled[...,1:]/np.maximum(1.e-9,1.-pooled[...,:1])
    hybrid=np.concatenate((full[...,:1],(1.-full[...,:1])*conditional),axis=-1)
    fits["conditional-mortality"]=(hybrid,value,coarse)
    metrics={k:stats(v[0],validation) for k,v in fits.items()}
    selected=min(metrics,key=lambda k:metrics[k]["nll"]);trans,value,coarse=fits[selected]
    terminal=np.ones((3,4,2,2,H))
    for i in range(3):
        valid=train&(parent==1)&(act==i)
        terminal[i]*=np.mean(end_act[valid]>i+1)
    vm=test&(parent==1)
    target_v=end_act[vm]>act[vm]+1
    dest=(act[vm],stage[vm],p[vm],green[vm],h[vm])
    report=dict(examples=len(a),games=len(np.unique(seed)),splits=[int(v.sum()) for v in (train,validation,test)],
                selection=metrics,shrink=selected,test=stats(trans,test),pooled_test=stats(pooled,test),
                value_brier=float(np.mean((value[dest]-target_v)**2)),
                act_only_value_brier=float(np.mean((terminal[dest]-target_v)**2)))
    write(out/"fit.json",report)
    write(out/"tables.json",dict(transition=trans.reshape(-1).tolist(),value=value.reshape(-1).tolist()))
    print(report,flush=True)


if __name__=="__main__":
    main()
