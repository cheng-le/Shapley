"""Resume 100k memmap-based Pile selection cells with strict JSON validation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from pile_main_100k.common import METHODS


def valid_cell(path: Path, method: str, seed: int, budget: int) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        cell = json.loads(path.read_text())
    except Exception:
        return False
    required = {
        "budget",
        "selected_corpora",
        "epochs",
        "primary_metric",
        "method",
        "seed",
        "accuracy",
        "val_loss",
        "score_cache",
        "token_cache",
    }
    if not required.issubset(cell):
        return False
    if cell["method"] != method or int(cell["seed"]) != seed or int(cell["budget"]) != budget:
        return False
    if int(cell["selected_corpora"]) != round(100000 * budget / 100):
        return False
    if "clean_seed" + str(seed) + ".npz" not in str(cell["score_cache"]):
        return False
    if str(cell["token_cache"]) != "pile_main_100k/token_cache":
        return False
    try:
        float(cell["accuracy"]); float(cell["val_loss"])
    except Exception:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-gpu", type=str, required=True)
    parser.add_argument("--method-start", type=int, required=True)
    parser.add_argument("--method-end", type=int, required=True)
    parser.add_argument("--budgets", nargs="+", type=int, default=[60, 80])
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    env = os.environ.copy()
    env.update({
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "MALLOC_ARENA_MAX": "2",
        "CUDA_VISIBLE_DEVICES": args.physical_gpu,
    })

    for budget in args.budgets:
        raw_dir = Path(f"pile_main_100k/selection_{budget}/raw_100k")
        raw_dir.mkdir(parents=True, exist_ok=True)
        for method_index in range(args.method_start, args.method_end + 1):
            method = METHODS[method_index]
            for seed in (0, 1, 2):
                output = raw_dir / f"method{method_index}_seed{seed}.json"
                log_path = output.with_suffix(".log")
                if valid_cell(output, method, seed, budget):
                    print(f"skip valid budget={budget} method{method_index} seed={seed}", flush=True)
                    continue
                cmd = [
                    sys.executable,
                    "-m",
                    "pile_main_100k.selection_accuracy_cached_runner",
                    "--gpu",
                    "0",
                    "--cache",
                    f"pile_main_100k/cache/clean_seed{seed}.npz",
                    "--token-cache",
                    "pile_main_100k/token_cache",
                    "--budget",
                    str(budget),
                    "--method",
                    method,
                    "--seed",
                    str(seed),
                    "--batch-size",
                    str(args.batch_size),
                    "--output",
                    str(output),
                ]
                print(f"run budget={budget} method{method_index} seed={seed} gpu={args.physical_gpu}", flush=True)
                with log_path.open("w") as log:
                    result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode != 0:
                    raise SystemExit(f"cell failed: {output}; see {log_path}")
                if not valid_cell(output, method, seed, budget):
                    raise SystemExit(f"cell produced invalid JSON: {output}")
                print(f"done budget={budget} method{method_index} seed={seed}", flush=True)


if __name__ == "__main__":
    main()
