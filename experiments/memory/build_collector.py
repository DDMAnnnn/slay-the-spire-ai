"""Build the isolated public-trace collector with GCC and a pinned sts_lightspeed checkout."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkout",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--cxx",default="g++")
    args=p.parse_args()
    root=args.checkout.resolve()
    rev=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()
    if rev!="7476a81954020087da31d41d16fddf475746ec2d":
        raise ValueError("Use the simulator revision in the experiment README.")
    out=args.output.resolve()
    objs=out.parent/"collector-objects"
    objs.mkdir(parents=True,exist_ok=True)
    flags=["-std=c++17","-O1","-DNDEBUG","-w","-include","algorithm",
           "-I"+str(root/"include"),"-I"+str(root/"json/single_include")]
    sources=sorted((root/"src").rglob("*.cpp"))+[Path(__file__).with_name("collect.cpp").resolve()]
    def compile_one(source):
        dest=objs/(source.stem+".o")
        subprocess.run([args.cxx,*flags,"-c",str(source),"-o",str(dest)],check=True)
        return str(dest)
    with ThreadPoolExecutor(max_workers=4) as pool:
        object_paths=list(pool.map(compile_one,sources))
    subprocess.run([args.cxx,*object_paths,"-o",str(out)],check=True)


if __name__=="__main__":
    main()
