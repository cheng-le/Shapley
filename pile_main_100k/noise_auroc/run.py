"""Compute noisy-corpus detection AUROC from a corrupted score cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from pile_main_100k.common import METHODS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results.json")
    args = parser.parse_args()
    data = np.load(args.cache, allow_pickle=False)
    labels = data["noise_labels"]
    if labels.sum() == 0:
        raise RuntimeError("Cache has no corrupted examples.")
    results = {
        method: float(roc_auc_score(labels, -data[f"score::{method}"].astype(np.float64)))
        for method in METHODS
    }
    payload = {"metric": "Noise AUROC", "noise_fraction": float(labels.mean()), "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
