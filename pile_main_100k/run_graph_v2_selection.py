"""Run the two Graph v2 methods without overwriting frozen raw cells."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from pile_main_100k.common import METHODS


def valid(path: Path, method: str, seed: int, budget: int) -> bool:
    try:
        cell = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        cell.get("method") == method
        and cell.get("seed") == seed
        and cell.get("budget") == budget
        and cell.get("selected_corpora") == budget * 1000
        and cell.get("selection", {}).get("strategy") == "symmetric_graph_greedy"
        and "graph_v2" in cell.get("score_cache", "")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-gpu", required=True)
    parser.add_argument("--budget", type=int, choices=[20, 40, 60, 80], required=True)
    args = parser.parse_args()
    output_dir = Path(f"pile_main_100k/selection_{args.budget}/raw_100k_graph_v2")
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": args.physical_gpu,
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "MALLOC_ARENA_MAX": "2",
    })
    for method_index in (10, 11):
        method = METHODS[method_index]
        for seed in range(3):
            output = output_dir / f"method{method_index}_seed{seed}.json"
            if valid(output, method, seed, args.budget):
                print(f"skip valid method={method_index} seed={seed}", flush=True)
                continue
            command = [
                sys.executable, "-m", "pile_main_100k.selection_accuracy_cached_runner",
                "--gpu", "0",
                "--cache", f"pile_main_100k/cache_graph_v2/clean_seed{seed}_graph_v2.npz",
                "--token-cache", "pile_main_100k/token_cache",
                "--budget", str(args.budget),
                "--method", method,
                "--seed", str(seed),
                "--batch-size", "16",
                "--graph-diversity",
                "--diversity-lambda", "0.05",
                "--output", str(output),
            ]
            log_path = output.with_suffix(".log")
            print(f"run budget={args.budget} method={method_index} seed={seed}", flush=True)
            with log_path.open("w") as log:
                result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode or not valid(output, method, seed, args.budget):
                raise SystemExit(f"failed: {output}; see {log_path}")
            print(f"done budget={args.budget} method={method_index} seed={seed}", flush=True)


if __name__ == "__main__":
    main()
