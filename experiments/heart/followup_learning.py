"""Fit a conditional logistic choice model on the visible card sets a prompt offers."""
import gzip
import hashlib
import json
from math import sqrt
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix

from .train import write

TASKS=25
CARDS=1+2*371
SIZE=11*5
WIDTH=CARDS+SIZE


def panel(data):
    row=[];col=[];val=[];starts=[];picked=[];first=[];tasks=[]
    n=0
    for _,task,ctx,options,y,b in data:
        starts.append(n);picked.append(n+y);first.append(n+b);tasks.append(task)
        base=task*WIDTH
        for cards in options:
            for c in cards:
                row.append(n);col.append(base+c);val.append(1/sqrt(max(1,len(cards))))
            size=base+CARDS+min(10,len(cards))*5
            for j,v in enumerate(ctx):
                row.append(n);col.append(size+j);val.append(v)
            n+=1
    x=csr_matrix((val,(row,col)),shape=(n,TASKS*WIDTH))
    start=np.array(starts);count=np.diff(np.r_[start,n])
    return x,start,np.array(picked),np.array(first),np.array(tasks),count


def metrics(p,data):
    x,starts,picks,first,tasks,counts=data
    z=x@p
    selected=np.array([i+z[i:i+n].argmax() for i,n in zip(starts,counts)])
    peak=np.maximum.reduceat(z,starts)
    lse=peak+np.log(np.add.reduceat(np.exp(z-np.repeat(peak,counts)),starts))
    def summarize(mask):
        return dict(n=int(mask.sum()),accuracy=float(np.mean(selected[mask]==picks[mask])),
                    first_legal=float(np.mean(first[mask]==picks[mask])),
                    nll=float(np.mean((lse-z[picks])[mask])))
    return dict(all=summarize(np.ones(len(tasks),bool)),tasks={str(i):summarize(tasks==i) for i in np.unique(tasks)})


def main():
    out=Path("build/public-followups")
    with gzip.open(out/"menus.jsonl.gz","rt") as fp:rows=list(map(json.loads,fp))
    split=[int.from_bytes(hashlib.sha256(str(v[0]).encode()).digest()[:4],"big")%10 for v in rows]
    groups=[panel([r for r,b in zip(rows,split) if keep(b)]) for keep in
            (lambda b:b<8,lambda b:b==8,lambda b:b==9)]
    tr,va,te=groups
    x,idx,y,_,_,counts=tr
    fits={};history={}
    for ridge in (.1,1.,8.,40.):
        def loss(coef):
            z=x@coef;m=np.maximum.reduceat(z,idx)
            exp=np.exp(z-np.repeat(m,counts));s=np.add.reduceat(exp,idx)
            p=exp/np.repeat(s,counts);p[y]-=1
            value=(m+np.log(s)-z[y]).sum()+.5*ridge*(coef@coef)
            return value,x.T@p+ridge*coef
        fitted=minimize(loss,np.zeros(TASKS*WIDTH),method="L-BFGS-B",jac=True,options={"maxiter":160,"ftol":1.e-8})
        fits[ridge]=fitted.x
        history[ridge]=dict(validation=metrics(fitted.x,va)["all"],iterations=fitted.nit)
    chosen=min(history,key=lambda r:history[r]["validation"]["nll"])
    weights=fits[chosen].reshape(TASKS,WIDTH)
    write(out/"public-menu.json",dict(card=weights[:,:CARDS].tolist(),size=weights[:,CARDS:].reshape(TASKS,11,5).tolist()))
    report=dict(splits=[len(g[1]) for g in groups],collection=json.loads((out/"collection.json").read_text()),
                history=history,selected=chosen,test=metrics(fits[chosen],te))
    write(out/"fit.json",report);print(report["splits"],chosen,report["test"]["all"],flush=True)


if __name__=="__main__":main()
