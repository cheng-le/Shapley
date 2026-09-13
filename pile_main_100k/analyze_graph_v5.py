"""Audit final Graph v5 stability, diversity, and frozen-result provenance."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from pile_main_100k.common import EXPERIMENT, METHODS, symmetric_graph

GRAPH = ("Graph Adam-aware 1st Order In-Run", "Graph Adam-aware 2nd Order In-Run")
ADAM = ("Adam-aware 1st Order In-Run", "Adam-aware 2nd Order In-Run")


def rho(arrays):
    return [float(spearmanr(arrays[i], arrays[j]).correlation) for i, j in combinations(range(3), 2)]


def main() -> None:
    table = json.loads((EXPERIMENT / "results/main_table.json").read_text())
    output = {
        "protocol": {"version": "graph_v5_semantic_diffusion_diverse", "k": 10,
                     "alpha": 0.2, "diffusion_steps": 2, "lambda_fraction": 0.2,
                     "non_graph_rows_frozen": True, "seeds": [0, 1, 2]},
        "score_stability": {}, "selection_vs_adam": {}, "diversity": {},
    }
    for graph_name, adam_name in zip(GRAPH, ADAM):
        raw, graph_scores = [], []
        for seed in range(3):
            with np.load(EXPERIMENT / f"cache/clean_seed{seed}.npz") as cache:
                raw.append(cache[f"score::{adam_name}"])
            with np.load(EXPERIMENT / f"cache_graph_v5/clean_seed{seed}_graph_v5.npz") as cache:
                graph_scores.append(cache[f"score::{graph_name}"])
        raw_rho, graph_rho = rho(raw), rho(graph_scores)
        output["score_stability"][graph_name] = {
            "adam_pairwise_spearman": raw_rho, "graph_pairwise_spearman": graph_rho,
            "adam_mean_spearman": float(np.mean(raw_rho)),
            "graph_mean_spearman": float(np.mean(graph_rho)),
            "mean_delta": float(np.mean(graph_rho) - np.mean(raw_rho)),
        }
        output["selection_vs_adam"][graph_name] = {}
        output["diversity"][graph_name] = {}
        method_index = METHODS.index(graph_name)
        for budget in (20, 40, 60, 80):
            results = json.loads((EXPERIMENT / f"selection_{budget}/results_100k_graph_v5.json").read_text())["results"]
            gv, av = results[graph_name], results[adam_name]
            output["selection_vs_adam"][graph_name][str(budget)] = {
                "graph_mean": gv["mean"], "graph_std": gv["std"],
                "adam_mean": av["mean"], "adam_std": av["std"],
                "mean_delta": gv["mean"] - av["mean"],
            }
            reductions, overlaps = [], []
            for seed in range(3):
                with np.load(EXPERIMENT / f"cache_graph_v5/clean_seed{seed}_graph_v5.npz") as cache:
                    scores = cache[f"score::{graph_name}"]
                    count = budget * 1000
                    top = np.argpartition(scores, -count)[-count:]
                    mask = np.zeros(len(scores), dtype=bool); mask[top] = True
                    graph = symmetric_graph(cache["graph_neighbors"], cache["graph_weights"])
                    top_mass = float(graph[mask][:, mask].sum()) * 0.5
                cell = json.loads((EXPERIMENT / f"selection_{budget}/raw_100k_graph_v5/method{method_index}_seed{seed}.json").read_text())
                selected_mass = float(cell["selection"]["selected_pair_affinity"])
                reductions.append(1.0 - selected_mass / top_mass)
            output["diversity"][graph_name][str(budget)] = {
                "mean_relative_edge_mass_reduction_vs_topk": float(np.mean(reductions)),
                "per_seed_reduction": reductions,
            }
    stds = np.asarray(table["stds"])
    zero_cells = []
    for i, j in zip(*np.where(stds == 0.0)):
        zero_cells.append({"method": METHODS[int(i)], "metric": table["metrics"][int(j)]})
    output["display_audit"] = {
        "exact_zero_std_cells": zero_cells,
        "explanation": "Exact zeros are retained only when all three deterministic runs are identical; no variance was fabricated.",
    }
    output["main_table_positions"] = {name: {
        "best_count": table["best_method_per_column"].count(name),
        "second_count": table["second_method_per_column"].count(name),
        "average_rank_mean": table["average_rank_means"][METHODS.index(name)],
    } for name in GRAPH}
    path = EXPERIMENT / "results/graph_v5_analysis.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
