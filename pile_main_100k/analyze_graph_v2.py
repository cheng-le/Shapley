"""Create a machine-readable audit of Graph v2 stability and diversity."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from pile_main_100k.common import EXPERIMENT


GRAPH_METHODS = (
    "Graph Adam-aware 1st Order In-Run",
    "Graph Adam-aware 2nd Order In-Run",
)
ADAM_METHODS = (
    "Adam-aware 1st Order In-Run",
    "Adam-aware 2nd Order In-Run",
)


def read(path: Path):
    return json.loads(path.read_text())


def pairwise_spearman(arrays):
    values = []
    for left, right in combinations(range(len(arrays)), 2):
        result = spearmanr(arrays[left], arrays[right])
        values.append(float(getattr(result, "statistic", result.correlation)))
    return values


def main() -> None:
    table = read(EXPERIMENT / "results" / "main_table.json")
    audit_rows = read(EXPERIMENT / "cache_graph_v2" / "diversity_audit.json")
    output = {
        "protocol": {
            "graph_version": "graph_v2_dynamic_diverse",
            "alpha": 0.2,
            "lambda_fraction": 0.05,
            "std_ddof": 0,
            "seeds": [0, 1, 2],
        },
        "score_stability": {},
        "selection_vs_adam": {},
        "selection_vs_static_graph": {},
        "diversity": {},
    }

    for graph_method, adam_method in zip(GRAPH_METHODS, ADAM_METHODS):
        adam_scores, graph_scores = [], []
        for seed in range(3):
            clean = np.load(EXPERIMENT / "cache" / f"clean_seed{seed}.npz", allow_pickle=False)
            graph = np.load(
                EXPERIMENT / "cache_graph_v2" / f"clean_seed{seed}_graph_v2.npz",
                allow_pickle=False,
            )
            adam_scores.append(clean[f"score::{adam_method}"])
            graph_scores.append(graph[f"score::{graph_method}"])
        adam_rho = pairwise_spearman(adam_scores)
        graph_rho = pairwise_spearman(graph_scores)
        output["score_stability"][graph_method] = {
            "adam_pairwise_spearman": adam_rho,
            "graph_pairwise_spearman": graph_rho,
            "adam_mean_spearman": float(np.mean(adam_rho)),
            "graph_mean_spearman": float(np.mean(graph_rho)),
            "mean_spearman_delta": float(np.mean(graph_rho) - np.mean(adam_rho)),
        }

        for budget in (20, 40, 60, 80):
            old = read(EXPERIMENT / f"selection_{budget}" / "results_100k.json")["results"]
            new = read(EXPERIMENT / f"selection_{budget}" / "results_100k_graph_v2.json")["results"]
            output["selection_vs_adam"].setdefault(graph_method, {})[str(budget)] = {
                "graph_mean": new[graph_method]["mean"],
                "graph_std": new[graph_method]["std"],
                "adam_mean": new[adam_method]["mean"],
                "adam_std": new[adam_method]["std"],
                "mean_delta": new[graph_method]["mean"] - new[adam_method]["mean"],
                "std_delta": new[graph_method]["std"] - new[adam_method]["std"],
            }
            output["selection_vs_static_graph"].setdefault(graph_method, {})[str(budget)] = {
                "v2_mean": new[graph_method]["mean"],
                "v2_std": new[graph_method]["std"],
                "static_mean": old[graph_method]["mean"],
                "static_std": old[graph_method]["std"],
                "mean_delta": new[graph_method]["mean"] - old[graph_method]["mean"],
                "std_delta": new[graph_method]["std"] - old[graph_method]["std"],
            }

    for method in GRAPH_METHODS:
        output["diversity"][method] = {}
        for budget in (20, 40, 60, 80):
            rows = [
                row for row in audit_rows
                if row["method"] == method and row["budget"] == budget
            ]
            output["diversity"][method][str(budget)] = {
                "mean_relative_reduction": float(np.mean([row["relative_reduction"] for row in rows])),
                "mean_topk_overlap": float(np.mean([row["topk_overlap"] for row in rows])),
                "mean_graph_coverage": float(np.mean([row["graph_coverage"] for row in rows])),
            }

    best, second = table["best_method_per_column"], table["second_method_per_column"]
    output["main_table_positions"] = {
        method: {"best_count": best.count(method), "second_count": second.count(method)}
        for method in GRAPH_METHODS
    }
    stds = np.asarray(table["stds"])
    display_zero = np.column_stack([
        stds[:, :2] < 0.00005,
        stds[:, 2:] < 0.0005,
    ])
    output["display_audit"] = {
        "exact_zero_std_cells": int((stds == 0).sum()),
        "selection_accuracy_is_percent": True,
        "rounded_zero_std_cells_at_display_precision": int(display_zero.sum()),
    }
    path = EXPERIMENT / "results" / "graph_v2_analysis.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
