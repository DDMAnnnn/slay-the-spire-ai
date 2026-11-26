"""Cross the simulator boundary only at public strategic decisions. Count real Heart entries."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
from pathlib import Path
import subprocess

import heart_native as engine
import numpy as np
import sts_env as sts

from .heart_curriculum import read_model
from .replay import apply_model
from .train import write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--early",type=int,default=384)
    p.add_argument("--late",type=int,default=2048);p.add_argument("--threads",type=int,default=4)
    p.add_argument("--runs",type=int,default=256);p.add_argument("--panel",type=int,default=261991)
    p.add_argument("--model",default="build/route-irl/selected-whole_map.json.gz")
    p.add_argument("--mode",type=int,default=-1);p.add_argument("--elite",type=int,default=0);p.add_argument("--boss",type=int,default=0)
    p.add_argument("--reference-combat",action="store_true",help="Pinned public search profile with eight samples and floor-dependent budgets")
    p.add_argument("--search-followups",action="store_true")
    p.add_argument("--handoff-act",type=int,choices=(3,4))
    p.add_argument("--neural-combat",type=Path,help="Use this locally trained student on zero-search-budget turns")
    args=p.parse_args()
    if args.neural_combat and (args.reference_combat or min(args.early,args.late)>0):
        p.error("the neural player needs at least one zero-budget combat phase")
    if args.search_followups and not args.reference_combat:
        p.error("follow-up search requires the reference combat profile")
    if args.runs<1 or args.runs>32768:raise ValueError("invalid requested panel size")
    out=Path("build/source-strategy")/f"{Path(args.model).name}-{args.panel}-{args.early}-{args.late}"
    if args.mode>=0 or args.elite or args.boss:out=out.with_name(out.name+f"-mode{args.mode}-elite{args.elite}-boss{args.boss}")
    if args.reference_combat:
        out=out.with_name(out.name+"-public8"+("-all" if args.search_followups else "-turn"))
    neural = None
    if args.neural_combat:
        import torch
        from .transfer_probe import EXAMPLE, model_code
        from threading import Lock
        torch.set_num_threads(1)
        device="cuda" if torch.cuda.is_available() else "cpu"
        neural,_=model_code().load(EXAMPLE/"weights.pt", device)
        neural.load_state_dict(torch.load(args.neural_combat,map_location=device,weights_only=True))
        neural.eval();inference_lock=Lock()
        out=out.with_name(out.name+"-"+args.neural_combat.parent.name+"-"+args.neural_combat.stem)
    prefixes = None
    if args.handoff_act:
        with gzip.open(f"build/source-handoff/act-{args.handoff_act}.json.gz","rt") as fp:
            prefixes=json.load(fp)
        out=out.with_name(out.name+f"-handoff{args.handoff_act}")
        args.runs=min(args.runs,len(prefixes))
    out.mkdir(parents=True,exist_ok=True)
    model=read_model(args.model);w=apply_model("first",model)
    ids=np.random.default_rng(args.panel).bit_generator.random_raw(args.runs).tolist()
    if prefixes:
        ids=[p["seed"] for p in prefixes[:args.runs]]
    def play(shard):
        cmd=[str(Path("build/public-expert-traces/python313/python.exe").resolve()),
             str(Path(__file__).with_name("strategic_source_worker.py")),"--early",str(args.early),"--late",str(args.late),"--mode",str(args.mode),"--elite",str(args.elite),"--boss",str(args.boss)]
        if neural is not None:cmd.append("--neural-combat")
        if args.reference_combat:cmd.append("--reference-combat")
        if args.search_followups:cmd.append("--search-followups")
        with subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,encoding="ascii",bufsize=1) as child:
            def send(v):
                child.stdin.write(json.dumps(v)+"\n");child.stdin.flush()
            try:
                for i in shard:
                    path=out/f"run-{i}.json"
                    if path.exists():
                        if json.loads(path.read_text())["seed"]!=str(ids[i]):raise ValueError("wrong cached seed")
                        continue
                    command={"seed":ids[i]}
                    if prefixes:
                        command.update(prefix=prefixes[i]["prefix"],state_hash=prefixes[i]["state_hash"])
                    send(command)
                    while True:
                        line=child.stdout.readline()
                        if not line:raise RuntimeError(f"source simulator closed on index {i}, seed {ids[i]}, exit {child.poll()}")
                        data=json.loads(line)
                        if "result" in data:
                            write(path,data["result"]);break
                        o=np.frombuffer(base64.b64decode(data["obs"]),np.float32)
                        mask=np.zeros(sts.N_ACTIONS,np.uint8);mask[data["legal"]]=1
                        if len(o)!=sts.OBS_SIZE:raise ValueError("wrong strategic observation width")
                        if data.get("neural"):
                            with inference_lock, torch.inference_mode():
                                x=torch.from_numpy(o.copy()).to(device)[None,:]
                                m=torch.from_numpy(mask.astype(bool)).to(device)[None,:]
                                pick=int(neural.dist(x,m).logits.argmax(1).item())
                        else:
                            pick=int(np.argmax(engine.scores(w,o,mask)))
                        if not mask[pick]:raise ValueError("illegal strategic action")
                        send(pick)
            finally:
                if child.poll() is None:
                    try:send({"stop":True})
                    except (BrokenPipeError,OSError):pass
                    try:child.stdin.close()
                    except (BrokenPipeError,OSError):pass
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        list(pool.map(play,[list(range(i,args.runs,args.threads)) for i in range(args.threads)]))
    games=[json.loads((out/f"run-{i}.json").read_text()) for i in range(args.runs)]
    result=dict(runs=len(games),early=args.early,late=args.late,mode=args.mode,elite=args.elite,boss=args.boss,panel_seed=args.panel,
                neural_checkpoint=str(args.neural_combat) if neural is not None else None,
                reference_combat=args.reference_combat,search_followups=args.search_followups,
                searched_followups=sum(v["menus"].get("searched_followup",0) for v in games),
                act2=sum(v["act"]>=2 for v in games),act3=sum(v["act"]>=3 for v in games),act4=sum(v["act"]>=4 for v in games),
                encounters=sum(v["heart"] for v in games),wins=sum(v["won"] for v in games),truncated=sum(v["capped"] for v in games),
                heart_encounter_seeds=[v["seed"] for v in games if v["heart"]],victory_seeds=[v["seed"] for v in games if v["won"]])
    if prefixes:
        pairs=list(zip(prefixes,games))
        result.update(conditional_start_act=args.handoff_act,reference_encounters=sum(p["reference_heart"] for p,_ in pairs),
                      reference_wins=sum(p["reference_win"] for p,_ in pairs),
                      gained=sum(g["heart"] and not p["reference_heart"] for p,g in pairs),
                      lost=sum(p["reference_heart"] and not g["heart"] for p,g in pairs))
    write(out/f"report-{args.runs}.json",result);print(result,flush=True)


if __name__=="__main__":
    main()
