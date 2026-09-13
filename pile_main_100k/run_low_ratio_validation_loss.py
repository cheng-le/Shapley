"""Run all methods for the 1/2/5/10% validation-loss experiment."""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from pile_main_100k.common import METHODS

BUDGETS=(1,2,5,10)

def valid(path, method, seed, budget):
    try: x=json.loads(path.read_text())
    except (OSError,json.JSONDecodeError): return False
    return (x.get("method")==method and x.get("seed")==seed and x.get("budget")==budget
            and x.get("epochs")==1 and x.get("eval_part")=="evaluation"
            and x.get("protocol_split","").endswith("protocol_v2_split.json"))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--physical-gpu",required=True)
    parser.add_argument("--methods",required=True,help="comma-separated method indices")
    args=parser.parse_args()
    env=os.environ.copy(); env.update({"CUDA_VISIBLE_DEVICES":args.physical_gpu,"TOKENIZERS_PARALLELISM":"false","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","MALLOC_ARENA_MAX":"2"})
    outdir=Path("pile_main_100k/low_ratio_validation_loss/raw"); outdir.mkdir(parents=True,exist_ok=True)
    for index in [int(x) for x in args.methods.split(",")]:
      method=METHODS[index]
      for seed in range(3):
       for budget in BUDGETS:
        out=outdir/f"method{index}_seed{seed}_budget{budget}.json"
        if valid(out,method,seed,budget):
            print(f"skip method={index} seed={seed} budget={budget}",flush=True); continue
        graph=index in (10,11)
        cache=(f"pile_main_100k/cache_graph_v2/clean_seed{seed}_graph_v2.npz" if graph else f"pile_main_100k/cache/clean_seed{seed}.npz")
        cmd=[sys.executable,"-m","pile_main_100k.selection_accuracy_cached_runner","--gpu","0","--cache",cache,"--token-cache","pile_main_100k/token_cache","--budget",str(budget),"--method",method,"--seed",str(seed),"--batch-size","16","--epochs","1","--protocol-split","pile_main_100k/protocol_v2_split.json","--eval-part","evaluation","--lr-schedule","constant","--output",str(out)]
        if graph: cmd.extend(["--graph-diversity","--diversity-lambda","0.05"])
        print(f"run method={index} seed={seed} budget={budget}",flush=True)
        with out.with_suffix(".log").open("w") as log: result=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode or not valid(out,method,seed,budget): raise SystemExit(f"failed {out}")
        print(f"done method={index} seed={seed} budget={budget}",flush=True)
if __name__=="__main__": main()
