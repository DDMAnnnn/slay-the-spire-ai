"""Prepare a pinned Windows replay runtime and its public observation layout."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

from .public_demonstrations import REV, DIGEST, URL
from .shared_combat import schema


def fetch(url, path, expected):
    path.parent.mkdir(exist_ok=True, parents=True)
    if not path.exists():
        tmp = path.with_suffix(path.suffix + ".tmp")
        with urllib.request.urlopen(url, timeout=60) as response:
            tmp.write_bytes(response.read())
        if hashlib.sha256(tmp.read_bytes()).hexdigest() != expected:
            raise ValueError(f"download checksum mismatch for {path.name}")
        tmp.replace(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"local checksum mismatch for {path.name}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("build/public-expert-traces"))
    p.add_argument("--intent-output",type=Path)
    p.add_argument("--python313",type=Path)
    args = p.parse_args()
    root = args.root
    base = f"https://raw.githubusercontent.com/SnyderConsulting/sts_ml/{REV}/agent/"
    manifest = root/"agent"/"manifest.json"
    fetch(base+"manifest.json",manifest,"fe11c950442206cfc7becf3a0408d173ce2678142008ce6a760bbc70999b0127")
    checks = json.loads(manifest.read_text())["files"]
    for name in ("runtime/sts_rl_env.py", "native/slaythespire.cp313-win_amd64.pyd"):
        fetch(base+name,root/"agent"/name,checks[name])
    fetch(URL,root/"replays.jsonl.gz",DIGEST)
    output = args.intent_output or root
    output.mkdir(parents=True, exist_ok=True)
    layout = schema()
    if args.intent_output:
        from .public_demo_intents import rendering_schema
        moves = root/"headers"/"MonsterMoves.h"
        fetch(f"https://raw.githubusercontent.com/SnyderConsulting/sts_ml/{REV}/include/constants/MonsterMoves.h",
              moves,"729e01caaa629401a2fb41f338c3f34446bf43538116477c7a255a132808f2a4")
        layout["rendering"] = rendering_schema(moves)
    (output/"layout.json").write_text(json.dumps(layout))
    if args.python313:
        cmd = [str(args.python313.resolve())]
    elif (root/"python313/python.exe").exists():
        cmd = [str((root/"python313/python.exe").resolve())]
    else:
        cmd = ["py","-3.13"]
    subprocess.run([*cmd,str(Path(__file__).with_name("public_demo_export.py")), "--root",str(output.resolve()),"--source-root",str(root.resolve())],check=True)


if __name__=="__main__":
    main()
