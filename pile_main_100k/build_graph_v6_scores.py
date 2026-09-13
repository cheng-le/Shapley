"""Build the label-free Table-5-style reciprocal Graph scores."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from pile_main_100k.common import reciprocal_conservative_graph_scores

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.cache, allow_pickle=False) as cache:
        neighbors, weights = cache["graph_neighbors"], cache["graph_weights"]
        first, agreement, reciprocal = reciprocal_conservative_graph_scores(
            cache["score::Adam-aware 1st Order In-Run"], neighbors, weights)
        second, _, _ = reciprocal_conservative_graph_scores(
            cache["score::Adam-aware 2nd Order In-Run"], neighbors, weights)
        payload = {
            "fingerprint": cache["fingerprint"], "graph_neighbors": neighbors,
            "graph_weights": weights, "graph_reciprocal_edges": reciprocal,
            "graph_agreement": agreement,
            "graph_metadata": json.dumps({
                "version": "graph_v6_reciprocal_conservative", "source_cache": str(args.cache),
                "k": int(neighbors.shape[1]), "smoothing": 0.2,
                "consistency_penalty": 1.0, "diversity_lambda_fraction": 0.05,
                "uses_domain_labels": False, "compatibility": "reciprocal_knn"}),
            "score::Graph Adam-aware 1st Order In-Run": first,
            "score::Graph Adam-aware 2nd Order In-Run": second,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print(f"saved={args.output}")
if __name__ == "__main__": main()
