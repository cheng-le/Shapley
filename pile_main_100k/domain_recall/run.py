"""Compute macro normalized domain Recall@1K for every attribution method."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from pile_case_study_common import bm25_scores
from pile_main_100k.graph_utils import gpu_chunked_knn
from pile_main_100k.common import (
    METHODS, confidence_gated_graph_weights, embeddings, load_rows, model,
    query_gradient, rank_scores, replay_graph_trajectory, seed_all,
    semantic_graph, tokenizer,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--k", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results.json")
    args = parser.parse_args()
    seed_all(args.seed)
    cache = np.load(args.cache, allow_pickle=False)
    gradients, directions, train_embeddings = cache["gradients"], cache["adam_directions"], cache["embeddings"]
    train_rows, val_rows = load_rows("train"), load_rows("val")
    by_domain = defaultdict(list)
    for row in val_rows:
        by_domain[row["domain"]].append(row)
    train_domains = np.asarray([row["domain"] for row in train_rows])
    eligible = [domain for domain, rows in by_domain.items() if len(rows) >= 5 and np.sum(train_domains == domain) >= 20]
    device = torch.device(f"cuda:{args.gpu}")
    net, tok = model(device), tokenizer()
    per_method = {method: {} for method in METHODS}
    documents = [" ".join(row["text"].split()[:256]) for row in train_rows]
    if "graph_neighbors" in cache and "graph_weights" in cache:
        neighbors, weights = cache["graph_neighbors"], cache["graph_weights"]
    else:
        neighbors, weights = gpu_chunked_knn(train_embeddings, device) if len(train_embeddings) > 20000 else semantic_graph(train_embeddings)
    gated_weights, confidence = confidence_gated_graph_weights(
        gradients, neighbors, weights
    )
    for domain in sorted(eligible):
        queries = by_domain[domain][:32]
        qgrad = query_gradient(net, tok, queries, device, limit=32)
        qembedding = embeddings(net, tok, queries, device).astype(np.float32).mean(0)
        qembedding /= np.linalg.norm(qembedding).clip(1e-8)
        scores = rank_scores(gradients, train_embeddings, qgrad, qembedding, args.seed, (neighbors, weights))
        sgd_trajectory = gradients.astype(np.float32) @ qgrad.astype(np.float32)
        adam_trajectory = directions.astype(np.float32) @ qgrad.astype(np.float32)
        scores["1st Order In-Run Data Shapley"] = sgd_trajectory
        scores["2nd Order In-Run Data Shapley"] = sgd_trajectory
        scores["Adam-aware 1st Order In-Run"] = adam_trajectory
        scores["Adam-aware 2nd Order In-Run"] = adam_trajectory
        scores["Graph Adam-aware 1st Order In-Run"] = replay_graph_trajectory(
            adam_trajectory, neighbors, gated_weights, confidence, args.seed, alpha=0.2
        )
        scores["Graph Adam-aware 2nd Order In-Run"] = replay_graph_trajectory(
            adam_trajectory, neighbors, gated_weights, confidence, args.seed,
            alpha=0.2, include_pair=True,
        )
        lexical_queries = [" ".join(row["text"].split()[:128]) for row in queries[:8]]
        scores["BM25"] = bm25_scores(documents, lexical_queries).mean(0)
        relevant = train_domains == domain
        denominator = min(args.k, int(relevant.sum()))
        for method, values in scores.items():
            selected = np.argpartition(values, -args.k)[-args.k:]
            per_method[method][domain] = float(relevant[selected].sum() / denominator)
        print(f"domain={domain} train={relevant.sum()} val={len(by_domain[domain])}", flush=True)
    results = {method: float(np.mean(list(values.values()))) for method, values in per_method.items()}
    payload = {
        "metric": f"Domain normalized Recall@{args.k}",
        "definition": "relevant items in top-K / min(K, number of relevant training items)",
        "eligibility": "train >= 20 and validation >= 5 corpora",
        "eligible_domains": sorted(eligible),
        "results": results,
        "per_domain": per_method,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
