"""Merge isolated memmap-based accuracy cells into one budget result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pile_main_100k.common import METHODS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results, sources = {}, []
    for method_index, method in enumerate(METHODS):
        accuracy_runs, loss_runs = [], []
        for seed in range(3):
            path = args.raw_dir / f"method{method_index}_seed{seed}.json"
            cell = json.loads(path.read_text())
            if cell["method"] != method or cell["seed"] != seed or cell["budget"] != args.budget:
                raise RuntimeError(f"Mismatched cell: {path}")
            accuracy_runs.append(float(cell["accuracy"])); loss_runs.append(float(cell["val_loss"])); sources.append(str(path))
        results[method] = {"runs": accuracy_runs, "mean": float(np.mean(accuracy_runs)), "std": float(np.std(accuracy_runs)),
                           "val_loss_runs": loss_runs, "val_loss_mean": float(np.mean(loss_runs)),
                           "val_loss_std": float(np.std(loss_runs))}
    payload = {"budget": args.budget, "selected_corpora": args.budget * 1000, "epochs": 1,
               "primary_metric": "validation_token_accuracy", "results": results, "raw_sources": sources}
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"merged={args.output} cells={len(sources)}")


if __name__ == "__main__":
    main()
