"""On-policy PPO from the existing Ironclad actor, with a trainable full-observation residual."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from time import perf_counter

import heart_native as engine
import numpy as np
import sts_env as sts
import torch
from torch.nn import functional as F

from .heart_curriculum import read_model
from .replay import apply_model
from .train import write
from .transfer_probe import EXAMPLE, model_code, evaluate


class Anchored:
    def __init__(self,device,threads,weights):
        self.device=device
        self.weights=weights
        self.threads=threads
        self.actor,_=model_code().load(EXAMPLE/"weights.pt",device)
        tuned=torch.load("build/ironclad-transfer/update-64.pt",map_location=device,weights_only=True)
        self.actor.load_state_dict(tuned["agent"])
        self.reference=copy.deepcopy(self.actor).eval().requires_grad_(False)
        # Keep the Ironclad representations. Fit the critic to this run's reward scale.
        torch.nn.init.zeros_(self.actor.v.weight)
        torch.nn.init.zeros_(self.actor.v.bias)

    def prior(self,obs,mask):
        logits=engine.prior_batch(self.weights,obs,mask.astype(np.uint8),self.threads)
        return torch.as_tensor(logits,device=self.device)

    def combine(self,distribution,reference,prior,mask):
        raw=distribution.logits-reference
        # Referencing the incumbent action removes arbitrary state-wise logit offsets.
        centered=raw-raw.gather(1,prior.argmax(1,keepdim=True))
        residual=8*torch.tanh(centered/8)
        return torch.distributions.Categorical(logits=(prior+residual).masked_fill(~mask,-1e9))

    def dist(self,obs,mask):
        # Used by the disjoint greedy evaluation, which supplies no hidden state.
        prior=self.prior(obs.cpu().numpy(),mask.cpu().numpy())
        fixed=self.reference.dist(obs,mask).logits
        return self.combine(self.actor.dist(obs,mask),fixed,prior,mask)


def shaping(obs):
    return .8*obs[:,0]/np.maximum(1,obs[:,1])+.16*obs[:,12:15].sum(1)


def identity_mask(obs,mask):
    return mask


@torch.inference_mode()
def evaluate_frozen(model,start,count,batch,threads):
    n=min(count,batch)
    env=sts.VecEnv(n,20,threads,3500,True,False,"ironclad",0.,0,0.)
    env.reset(list(range(start,start+n)))
    seen=np.zeros(n,bool);records={};uncertain=decisions=0
    m,_,_=sts.OBS_LAYOUT["monsters"]
    heart=sts.MONSTER_NAMES.index("CORRUPT_HEART")+1
    begin=perf_counter()
    while len(records)<count:
        seed=env.seed.copy();active=seed<start+count
        obs=env.obs;mask=env.mask
        seen|=active&(obs[:,4]==4)&(obs[:,7]>0)&(obs[:,m]==heart)&(obs[:,m+6]>0)
        moves=mask.argmax(1).astype(np.int32)
        ids=np.flatnonzero(active&(mask.sum(1)>1))
        if len(ids):
            prior=engine.prior_batch(model.weights,obs[ids],mask[ids],threads)
            greedy=prior.argmax(1);moves[ids]=greedy
            # Each correction is at most eight and the incumbent has exactly zero.
            # If its margin exceeds eight, neural inference cannot change the argmax.
            best_two=np.partition(prior,-2,axis=1)[:,-2:]
            use=best_two.max(1)-best_two.min(1)<=8.
            decisions+=len(ids);uncertain+=int(use.sum())
            ids=ids[use]
            if len(ids):
                o=torch.as_tensor(obs[ids],device=model.device)
                legal=torch.as_tensor(mask[ids],dtype=torch.bool,device=model.device)
                fixed=model.reference.dist(o,legal).logits
                predicted=model.combine(model.actor.dist(o,legal),fixed,
                            torch.as_tensor(prior[use],device=model.device),legal)
                moves[ids]=predicted.logits.argmax(1).cpu().numpy()
        if env.step(moves): raise RuntimeError("invalid greedy action")
        done=env.terminated.astype(bool)|env.truncated.astype(bool)
        for j in np.flatnonzero(done&active):
            end=env.final_obs[j]
            records[int(seed[j])]=[int(end[3]),int(end[4]),int(seen[j]),int(env.won[j]),int(env.truncated[j])]
        seen[done]=False
    if sorted(records)!=list(range(start,start+count)): raise RuntimeError("incomplete seed range")
    rows=np.asarray([records[i] for i in sorted(records)])
    return dict(start=start,runs=count,mean_floor=float(rows[:,0].mean()),act2=int((rows[:,1]>=2).sum()),
                act3=int((rows[:,1]>=3).sum()),act4=int((rows[:,1]>=4).sum()),
                heart_encounter_seeds=(start+np.flatnonzero(rows[:,2])).tolist(),
                victory_seeds=(start+np.flatnonzero(rows[:,3])).tolist(),truncated=int(rows[:,4].sum()),
                neural_decisions=uncertain,free_choices=decisions,seconds=round(perf_counter()-begin,2))


def evaluate_panel(model,args):
    digest=hashlib.sha256(args.evaluate.read_bytes()).hexdigest()
    path=args.out/f"panel-{args.start}-{args.count}.json"
    records=[]
    if path.exists():
        previous=json.loads(path.read_text(encoding="utf8"))
        if previous.get("checkpoint_sha256")!=digest:
            raise ValueError("a different checkpoint already has a report for this panel")
        records=previous["shards"]
    done=sum(s["runs"] for s in records)
    for offset in range(done,args.count,2048):
        s=evaluate_frozen(model,args.start+offset,min(2048,args.count-offset),args.batch,args.threads)
        records.append(s)
        result={k:sum(r[k] for r in records) for k in ("runs","act2","act3","act4","truncated","free_choices","neural_decisions","seconds")}
        result.update(start=args.start,planned=args.count,checkpoint_sha256=digest,shards=records,
                      mean_floor=sum(r["runs"]*r["mean_floor"] for r in records)/result["runs"],
                      heart_encounter_seeds=[seed for r in records for seed in r["heart_encounter_seeds"]],
                      victory_seeds=[seed for r in records for seed in r["victory_seeds"]])
        write(path,result)
        print(result["runs"],result["act3"],len(result["heart_encounter_seeds"]),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--updates",type=int,default=256)
    p.add_argument("--batch",type=int,default=128)
    p.add_argument("--horizon",type=int,default=32)
    p.add_argument("--threads",type=int,default=8)
    p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out",type=Path,default=Path("build/anchored-ppo"))
    p.add_argument("--evaluate",type=Path)
    p.add_argument("--start",type=int,default=189000000)
    p.add_argument("--count",type=int,default=32768)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(18611);torch.set_num_threads(3)
    rng=np.random.default_rng(18611)
    torch.distributions.Distribution.set_default_validate_args(False)
    parent=read_model("build/trajectory-teacher/challenger.json.gz")
    w=apply_model("first",parent)
    learner=Anchored(args.device,args.threads,w)
    if args.evaluate:
        learner.actor.load_state_dict(torch.load(args.evaluate,map_location=args.device,weights_only=True))
        evaluate_panel(learner,args)
        return
    n=args.batch
    env=sts.VecEnv(n,20,args.threads,3500,True,False,"ironclad",0.,0,0.)
    env.reset(list(range(186000000,186000000+n)))
    optimizer=torch.optim.AdamW(learner.actor.parameters(),lr=.00007,weight_decay=.01)
    report=dict(start=186000000,validation_start=187000000,batch=n,horizon=args.horizon,updates=[],validation=[])
    m,_,_=sts.OBS_LAYOUT["monsters"]
    heart=sts.MONSTER_NAMES.index("CORRUPT_HEART")+1
    encountered=np.zeros(n,bool)
    best=-1
    for update in range(1,args.updates+1):
        data={k:[] for k in ("obs","mask","prior","fixed","actions","logp","value","reward","stop")}
        counts=np.zeros(5,int)
        for t in range(args.horizon):
            obs=env.obs.copy();mask=env.mask.astype(bool)
            o=torch.as_tensor(obs,device=args.device);legal=torch.as_tensor(mask,device=args.device)
            with torch.no_grad():
                prior=learner.prior(obs,mask)
                fixed=learner.reference.dist(o,legal).logits
                original,value=learner.actor.dist_value(o,legal)
                dist=learner.combine(original,fixed,prior,legal)
                if update==1 and t==0:
                    native=engine.teacher_batch(w,obs,mask.astype(np.uint8),args.threads)
                    if not np.array_equal(dist.logits.argmax(1).cpu().numpy(),native):
                        raise RuntimeError("initial greedy policy differs from the incumbent")
                action=dist.sample()
                logp=dist.log_prob(action)
                counts[3]+=int((action!=prior.argmax(1)).sum())
                counts[4]+=int((legal.sum(1)>1).sum())
            a=action.cpu().numpy().astype(np.int32)
            if env.step(a): raise RuntimeError("illegal sampled action")
            done=env.terminated.astype(bool)|env.truncated.astype(bool)
            nxt=env.obs.copy();nxt[done]=env.final_obs[done]
            seen=(nxt[:,4]==4)&(nxt[:,7]>0)&(nxt[:,m]==heart)&(nxt[:,m+6]>0)
            entered=seen & ~encountered
            encountered=(encountered|seen)&~done
            won=env.won.astype(bool)
            reward=.12*np.maximum(0,nxt[:,3]-obs[:,3])+.65*np.maximum(0,nxt[:,4]-obs[:,4])
            reward+=.12*((obs[:,7]>0)&(nxt[:,7]==0)&(nxt[:,0]>0))+12*entered+4*won-3*(done&~won)
            reward+=.995*shaping(nxt)*(~done)-shaping(obs)
            counts[:3]+=[done.sum(),entered.sum(),won.sum()]
            for name,v in (("obs",obs),("mask",mask),("prior",prior.cpu().numpy()),("fixed",fixed.cpu().numpy()),
                           ("actions",a),("logp",logp.cpu().numpy()),("value",value.cpu().numpy()),
                           ("reward",reward),("stop",done)):
                data[name].append(v)
        values=np.stack(data["value"])
        adv=np.zeros_like(values)
        with torch.no_grad():
            tail=learner.actor.value(torch.as_tensor(env.obs,device=args.device)).cpu().numpy()
        gae=np.zeros(n)
        for t in reversed(range(args.horizon)):
            keep=~data["stop"][t]
            delta=data["reward"][t]+.995*tail*keep-values[t]
            gae=delta+.995*.95*gae*keep
            adv[t]=gae;tail=values[t]
        data["returns"]=list(adv+values)
        data["adv"]=list((adv-adv.mean())/max(.001,adv.std()))
        arrays={k:np.concatenate(v) for k,v in data.items()}
        for epoch in range(2):
            for ix in np.array_split(rng.permutation(len(arrays["obs"])),max(1,len(arrays["obs"])//256)):
                def tensor(k):return torch.as_tensor(arrays[k][ix],device=args.device)
                mask=tensor("mask");prior=tensor("prior")
                raw,val=learner.actor.dist_value(tensor("obs"),mask)
                dist=learner.combine(raw,tensor("fixed"),prior,mask)
                ratio=(dist.log_prob(tensor("actions"))-tensor("logp")).exp()
                advantage=tensor("adv")
                policy=-torch.minimum(ratio*advantage,ratio.clamp(.8,1.2)*advantage).mean()
                kl=(dist.probs*(dist.logits-prior.log_softmax(1))).sum(1).mean()
                loss=policy+.5*F.smooth_l1_loss(val,tensor("returns"))+.015*kl-.006*dist.entropy().mean()
                optimizer.zero_grad();loss.backward()
                torch.nn.utils.clip_grad_norm_(learner.actor.parameters(),.7);optimizer.step()
        report["updates"].append(dict(update=update,episodes=int(counts[0]),entries=int(counts[1]),wins=int(counts[2]),
                    sampled_changes=int(counts[3]),free_choices=int(counts[4]),kl=float(kl.detach())))
        if update in (16,48,96,160,256,args.updates):
            result=evaluate(learner,identity_mask,args.device,"all",parent,187000000,512,n,args.threads,"ironclad")
            report["validation"].append(dict(update=update,**result))
            rank=result["act3"]+5*result["act4"]+10*len(result["heart_encounter_seeds"])+.04*result["act2"]
            if rank>best:
                best=rank;report["selected_update"]=update
                torch.save(learner.actor.state_dict(),args.out/"selected.pt")
            print(update,result,flush=True)
        write(args.out/"report.json",report)


if __name__=="__main__":
    main()
