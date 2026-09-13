"""Merge process-isolated token-accuracy selection cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    args = parser.parse_args()
    results, sources = {}, []
    for method_index, method in enumerate(args.methods):
        accuracy_runs, loss_runs = [], []
        for seed in (0, 1, 2):
            path = args.raw_dir / f"method{method_index}_seed{seed}.json"
            cell = json.loads(path.read_text())["results"][method]
            accuracy_runs.append(float(cell["runs"][0]))
            loss_runs.append(float(cell["val_loss_runs"][0]))
            sources.append(str(path))
        results[method] = {
            "runs": accuracy_runs, "mean": float(np.mean(accuracy_runs)), "std": float(np.std(accuracy_runs)),
            "val_loss_runs": loss_runs, "val_loss_mean": float(np.mean(loss_runs)), "val_loss_std": float(np.std(loss_runs)),
        }
    payload = {
        "budget": args.budget, "selected_corpora": args.budget * 1000, "epochs": 1,
        "primary_metric": "validation_token_accuracy", "results": results, "raw_sources": sources,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
