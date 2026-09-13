"""Build reusable clean or corrupted GPT-2 attribution caches on one GPU."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from pile_case_study_common import bm25_scores
from pile_main_100k.common import (
    corrupt_rows, dataset_fingerprint, embeddings, graph_score, load_rows, model,
    query_gradient, rank_scores, seed_all, semantic_graph, stratified_sample_rows, tokenizer,
)
from pile_main_100k.graph_utils import gpu_chunked_knn
from pile_main_100k.trajectory import trajectory_attribution


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seed_all(args.seed)
    device = torch.device(f"cuda:{args.gpu}")
    train_rows, val_rows = load_rows("train"), load_rows("val")
    noise_labels = np.zeros(len(train_rows), dtype=np.int8)
    if args.noise:
        train_rows, noise_labels = corrupt_rows(train_rows, args.noise, args.seed + 991)
    net, tok = model(device), tokenizer()
    train_embeddings = embeddings(net, tok, train_rows, device)
    utility_rows = stratified_sample_rows(val_rows, 64, seed=2027)
    query_embedding = embeddings(net, tok, utility_rows, device).astype(np.float32).mean(0)
    query_embedding /= np.linalg.norm(query_embedding).clip(1e-8)
    qgrad = query_gradient(net, tok, utility_rows, device)
    grads, directions, losses, trajectory_scores = trajectory_attribution(
        net, tok, train_rows, val_rows, device, args.seed
    )
    neighbors, weights = gpu_chunked_knn(train_embeddings, device) if len(train_embeddings) > 20000 else semantic_graph(train_embeddings)
    scores = rank_scores(grads, train_embeddings, qgrad, query_embedding, args.seed, (neighbors, weights))
    scores.update(trajectory_scores)
    scores["Graph Adam-aware 1st Order In-Run"] = graph_score(
        trajectory_scores["Adam-aware 1st Order In-Run"], grads, neighbors, weights
    )
    scores["Graph Adam-aware 2nd Order In-Run"] = graph_score(
        trajectory_scores["Adam-aware 2nd Order In-Run"], grads, neighbors, weights
    )
    documents = [" ".join(row["text"].split()[:256]) for row in train_rows]
    queries = [" ".join(row["text"].split()[:128]) for row in val_rows[:16]]
    scores["BM25"] = bm25_scores(documents, queries).mean(0).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        fingerprint=dataset_fingerprint(load_rows("train")),
        gradients=grads,
        adam_directions=directions,
        embeddings=train_embeddings,
        graph_neighbors=neighbors,
        graph_weights=weights,
        losses=losses,
        noise_labels=noise_labels,
        domains=np.asarray([row["domain"] for row in train_rows]),
        **{f"score::{name}": value for name, value in scores.items()},
    )
    print(f"saved={args.output}", flush=True)


if __name__ == "__main__":
    main()
