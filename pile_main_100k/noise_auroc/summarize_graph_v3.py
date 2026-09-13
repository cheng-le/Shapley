"""Recompute only Graph v3 noise AUROC rows and preserve frozen baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from pile_main_100k.common import conservative_gradient_graph_scores

GRAPH = (
    ("Graph Adam-aware 1st Order In-Run", "Adam-aware 1st Order In-Run"),
    ("Graph Adam-aware 2nd Order In-Run", "Adam-aware 2nd Order In-Run"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--caches", type=Path, nargs="+", required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.base.read_text())
    details = payload["details"]
    fractions = []
    for graph_name, adam_name in GRAPH:
        runs = []
        for path in args.caches:
            with np.load(path, allow_pickle=False) as cache:
                labels = cache["noise_labels"]
                fractions.append(float(labels.mean()))
                score, _, _ = conservative_gradient_graph_scores(
                    cache[f"score::{adam_name}"], cache["gradients"],
                    cache["graph_neighbors"], cache["graph_weights"], 0.2,
                )
                runs.append(float(roc_auc_score(labels, -score.astype(np.float64))))
        details[graph_name] = {
            "runs": runs, "mean": float(np.mean(runs)), "std": float(np.std(runs)),
        }
        payload["results"][graph_name] = details[graph_name]["mean"]
    payload["noise_fraction"] = float(np.mean(fractions))
    payload["source_caches"] = [str(path) for path in args.caches]
    payload["graph_version"] = {
        "name": "graph_v3_conservative_diverse", "smoothing": 0.2,
        "base_results": str(args.base),
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({name: details[name] for name, _ in GRAPH}, indent=2))


if __name__ == "__main__":
    main()
