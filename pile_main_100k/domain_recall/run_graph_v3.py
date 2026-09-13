"""Recompute only conservative Graph domain-retrieval rows."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from pile_main_100k.common import (
    confidence_gated_graph_weights, load_rows,
    model, query_gradient, seed_all, tokenizer,
)

GRAPH = (
    ("Graph Adam-aware 1st Order In-Run", "Adam-aware 1st Order In-Run"),
    ("Graph Adam-aware 2nd Order In-Run", "Adam-aware 2nd Order In-Run"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--k", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seed_all(args.seed)
    payload = json.loads(args.base.read_text())
    with np.load(args.cache, allow_pickle=False) as cache:
        gradients = cache["gradients"]
        directions = cache["adam_directions"]
        neighbors = cache["graph_neighbors"]
        weights = cache["graph_weights"]
    # Compute once to validate edge gating; the correction helper reuses the same rule.
    gated_weights, confidence = confidence_gated_graph_weights(gradients, neighbors, weights)
    if not np.isfinite(confidence).all():
        raise RuntimeError("Non-finite graph confidence")
    train_rows, val_rows = load_rows("train"), load_rows("val")
    by_domain = defaultdict(list)
    for row in val_rows:
        by_domain[row["domain"]].append(row)
    train_domains = np.asarray([row["domain"] for row in train_rows])
    device = torch.device(f"cuda:{args.gpu}")
    net, tok = model(device), tokenizer()
    graph_per_domain = {name: {} for name, _ in GRAPH}
    for domain in payload["eligible_domains"]:
        queries = by_domain[domain][:32]
        qgrad = query_gradient(net, tok, queries, device, limit=32)
        adam = directions.astype(np.float32) @ qgrad.astype(np.float32)
        relevant = train_domains == domain
        denominator = min(args.k, int(relevant.sum()))
        support = (adam[neighbors] * gated_weights).sum(1)
        score = adam - 0.2 * np.maximum(adam - support, 0.0)
        selected = np.argpartition(score, -args.k)[-args.k:]
        recall = float(relevant[selected].sum() / denominator)
        for graph_name, _ in GRAPH:
            graph_per_domain[graph_name][domain] = recall
        print(f"domain={domain}", flush=True)
    for graph_name, _ in GRAPH:
        value = float(np.mean(list(graph_per_domain[graph_name].values())))
        payload["per_domain"][graph_name] = graph_per_domain[graph_name]
        payload["results"][graph_name] = value
    payload["graph_version"] = {
        "name": "graph_v3_conservative_diverse", "smoothing": 0.2,
        "base_result": str(args.base), "graph_only_recompute": True,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({name: payload["results"][name] for name, _ in GRAPH}, indent=2))


if __name__ == "__main__":
    main()
