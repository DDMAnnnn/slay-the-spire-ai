"""Adapt an upstream Defect policy with Ironclad demonstrations and on-policy PPO."""
import argparse
from pathlib import Path
from time import perf_counter

import heart_native as native
import numpy as np
import sts_env as sts
import torch

from .heart_curriculum import read_model
from .replay import apply_model
from .train import write
from .transfer_probe import EXAMPLE,evaluate,model_code


def teacher_targets(obs,actions):
    n=len(actions)
    target=np.zeros((n,sts.N_ACTIONS),dtype=bool)
    target[np.arange(n),actions]=True
    # Two hand slots can represent the same play on the same target.
    off,size,width=sts.OBS_LAYOUT["hand"]
    hand=obs[:,off:off+size*width].reshape(n,size,width)
    play=np.flatnonzero(actions<sts.ACT_POTION)
    if len(play):
        source=hand[play,actions[play]//5]
        same=(hand[play,:,:3]==source[:,None,:3]).all(2)
        batch,slot=np.where(same)
        target[play[batch],5*slot+actions[play[batch]]%5]=True
    return target


def train(args):
    torch.manual_seed(15626)
    rng=np.random.default_rng(15626)
    torch.set_num_threads(3)
    torch.distributions.Distribution.set_default_validate_args(False)
    device=args.device
    src=model_code()
    actor,_=src.load(EXAMPLE/"weights.pt",device)
    parent=read_model("build/trajectory-teacher/challenger.json.gz")
    weights=apply_model("first",parent)
    n=args.batch
    env=sts.VecEnv(n,20,args.threads,3500,True,False,"ironclad",0.,0,0.)
    env.reset(list(range(156000000,156000000+n)))
    opt=torch.optim.AdamW(actor.parameters(),lr=.00016)
    args.out.mkdir(exist_ok=True,parents=True)
    report=dict(start=156000000,batch=n,horizon=args.horizon,device=device,validation_start=157000000,
                source_control="build/transfer-probe/zero-shot.json",checkpoints=[],updates=[])
    offset,_,width=sts.OBS_LAYOUT["monsters"]
    heart=sts.MONSTER_NAMES.index("CORRUPT_HEART")+1
    last_heart=np.zeros(n,dtype=bool)
    clock=perf_counter()
    for update in range(1,args.cloning+args.ppo+1):
        cloning=update<=args.cloning
        if update==args.cloning+1:
            opt=torch.optim.AdamW(actor.parameters(),lr=.000045)
        data={k:[] for k in ("obs","mask","target","action","logp","value","reward","stop")}
        teacher_agree=decisions=entered=0
        actor.train()
        for step in range(args.horizon):
            obs=np.asarray(env.obs).copy()
            legal=np.asarray(env.mask).copy()
            o=torch.as_tensor(obs,device=device)
            with torch.no_grad():
                mask=src.no_unpick(o,torch.as_tensor(legal,dtype=torch.bool,device=device))
                dist,val=actor.dist_value(o,mask)
                pred=dist.logits.argmax(1) if cloning else dist.sample()
                actions=pred.cpu().numpy().copy()
                logp=dist.log_prob(pred).cpu().numpy()
            allowed=mask.cpu().numpy()
            reference=native.teacher_batch(weights,obs,allowed.astype(np.uint8),args.threads)
            target=teacher_targets(obs,reference)&allowed
            free=allowed.sum(1)>1
            teacher_agree+=int((target[np.arange(n),actions]&free).sum())
            decisions+=int(free.sum())
            if cloning:
                rollin=.10+.55*update/max(args.cloning,1)
                use_teacher=rng.random(n)>rollin
                actions[use_teacher]=reference[use_teacher]
            if env.step(actions.astype(np.int32)):
                raise RuntimeError("illegal training action")
            stop=np.asarray(env.terminated,dtype=bool)|np.asarray(env.truncated,dtype=bool)
            nxt=np.asarray(env.obs).copy()
            nxt[stop]=np.asarray(env.final_obs)[stop]
            seen=(nxt[:,4]==4)&(nxt[:,7]>0)&(nxt[:,offset]==heart)&(nxt[:,offset+6]>0)
            new=seen & ~last_heart
            entered+=int(new.sum())
            last_heart=(last_heart|seen)&~stop
            # Each change is observed after the action, including the final pre-reset observation.
            reward=.08*(nxt[:,3]-obs[:,3])+.8*(nxt[:,4]-obs[:,4])+.12*(nxt[:,12:15]-obs[:,12:15]).sum(1)
            reward+=.08*((obs[:,7]>0)&(nxt[:,7]==0)&(nxt[:,0]>0))+4*new+np.asarray(env.won)
            for key,value in (("obs",obs),("mask",allowed),("target",target),("action",actions),
                              ("logp",logp),("value",val.detach().cpu().numpy()),("reward",reward),("stop",stop)):
                data[key].append(value)
        if not cloning:
            with torch.no_grad():
                tail=actor.value(torch.as_tensor(np.asarray(env.obs),device=device)).cpu().numpy()
            prior=np.stack(data["value"])
            adv=np.zeros_like(prior)
            gae=np.zeros(n)
            for t in reversed(range(args.horizon)):
                keep=~data["stop"][t]
                delta=data["reward"][t]+.995*tail*keep-prior[t]
                gae=delta+.995*.95*keep*gae
                adv[t]=gae
                tail=prior[t]
            data["returns"]=list(adv+prior)
            data["adv"]=list((adv-adv.mean())/max(adv.std(),.001))
        arrays={k:np.concatenate(v) for k,v in data.items()}
        # Forced actions contribute to the value function. Imitation uses genuine choices.
        indices=np.flatnonzero(arrays["mask"].sum(1)>1) if cloning else np.arange(len(arrays["obs"]))
        losses=[]
        for epoch in range(2):
            for ix in np.array_split(rng.permutation(indices),max(1,int(np.ceil(len(indices)/256)))):
                def tensor(key):
                    return torch.as_tensor(arrays[key][ix],device=device)
                dist,values=actor.dist_value(tensor("obs"),tensor("mask"))
                imitation=-dist.logits.masked_fill(~tensor("target"),-1e9).logsumexp(1).mean()
                if cloning:
                    loss=imitation
                else:
                    ratio=(dist.log_prob(tensor("action"))-tensor("logp")).exp()
                    advantage=tensor("adv")
                    policy=-torch.minimum(ratio*advantage,ratio.clamp(.8,1.2)*advantage).mean()
                    vloss=torch.nn.functional.smooth_l1_loss(values,tensor("returns"))
                    loss=policy+.5*vloss-.01*dist.entropy().mean()+.12*imitation
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(),.7)
                opt.step()
                losses.append(float(loss.detach()))
        record=dict(update=update,stage="cloning" if cloning else "ppo",rollout_steps=n*args.horizon,
                    teacher_agreement=teacher_agree/max(1,decisions),training_heart_entries=entered,
                    loss=float(np.mean(losses)),seconds=round(perf_counter()-clock,1))
        report["updates"].append(record)
        print(record,flush=True)
        if update in {max(1,args.cloning//2),args.cloning,args.cloning+max(1,args.ppo//2),args.cloning+args.ppo}:
            path=args.out/f"update-{update}.pt"
            torch.save({"agent":actor.state_dict()},path)
            actor.eval()
            summary=evaluate(actor,src.no_unpick,device,"all",parent,157000000,args.validate,n,args.threads,"ironclad")
            report["checkpoints"].append(dict(update=update,path=path.name,**summary))
            print("validation",update,summary,flush=True)
        write(args.out/"report.json",report)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=Path("build/ironclad-transfer"))
    p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--cloning",type=int,default=32)
    p.add_argument("--ppo",type=int,default=32)
    p.add_argument("--batch",type=int,default=128)
    p.add_argument("--horizon",type=int,default=32)
    p.add_argument("--validate",type=int,default=256)
    p.add_argument("--threads",type=int,default=6)
    train(p.parse_args())
