"""Aggregate all noisy-cache repetitions while preserving each raw AUROC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from pile_main_100k.common import METHODS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--caches", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = {method: [] for method in METHODS}
    fractions = []
    for path in args.caches:
        cache = np.load(path, allow_pickle=False)
        labels = cache["noise_labels"]
        fractions.append(float(labels.mean()))
        for method in METHODS:
            raw[method].append(float(roc_auc_score(labels, -cache[f"score::{method}"].astype(np.float64))))
    details = {
        method: {"runs": values, "mean": float(np.mean(values)), "std": float(np.std(values))}
        for method, values in raw.items()
    }
    payload = {
        "metric": "Noise AUROC",
        "noise_fraction": float(np.mean(fractions)),
        "num_runs": len(args.caches),
        "results": {method: values["mean"] for method, values in details.items()},
        "details": details,
        "source_caches": [str(path) for path in args.caches],
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
