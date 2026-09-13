"""Independent-tuning pilot for low selection ratios."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

METHODS = (
    ("Random", "pile_main_100k/cache/clean_seed{seed}.npz", False),
    ("Adam-aware 1st Order In-Run", "pile_main_100k/cache/clean_seed{seed}.npz", False),
    ("Graph Adam-aware 1st Order In-Run", "pile_main_100k/cache_graph_v2/clean_seed{seed}_graph_v2.npz", True),
)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--physical-gpu", required=True)
    parser.add_argument("--budget", type=int, choices=[5,10,15,20], required=True)
    parser.add_argument("--seeds", default="0")
    args=parser.parse_args()
    env=os.environ.copy(); env.update({"CUDA_VISIBLE_DEVICES":args.physical_gpu,"TOKENIZERS_PARALLELISM":"false","OMP_NUM_THREADS":"1"})
    outdir=Path("pile_main_100k/ratio_tuning_pilot"); outdir.mkdir(exist_ok=True)
    for seed in [int(value) for value in args.seeds.split(",")]:
      for index,(method,cache_template,graph) in enumerate(METHODS):
        out=outdir/f"budget{args.budget}_method{index}_seed{seed}.json"
        if out.exists():
            print(f"skip budget={args.budget} method={method} seed={seed}", flush=True)
            continue
        cache=cache_template.format(seed=seed)
        cmd=[sys.executable,"-m","pile_main_100k.selection_accuracy_cached_runner","--gpu","0","--cache",cache,"--token-cache","pile_main_100k/token_cache","--budget",str(args.budget),"--method",method,"--seed",str(seed),"--batch-size","16","--epochs","2","--protocol-split","pile_main_100k/protocol_v2_split.json","--eval-part","valuation","--lr-schedule","cosine","--output",str(out)]
        if graph: cmd.extend(["--graph-diversity","--diversity-lambda","0.05"])
        log=out.with_suffix(".log")
        print(f"run budget={args.budget} method={method} seed={seed}",flush=True)
        with log.open("w") as handle: result=subprocess.run(cmd,env=env,stdout=handle,stderr=subprocess.STDOUT)
        if result.returncode: raise SystemExit(f"failed {out}")
        print(f"done budget={args.budget} method={method} seed={seed}",flush=True)
if __name__=="__main__": main()
