"""Build conservative, gradient-gated Graph scores from a frozen cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pile_main_100k.common import conservative_gradient_graph_scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoothing", type=float, default=0.2)
    args = parser.parse_args()

    with np.load(args.cache, allow_pickle=False) as cache:
        gradients = cache["gradients"]
        neighbors = cache["graph_neighbors"]
        weights = cache["graph_weights"]
        first, gated_weights, confidence = conservative_gradient_graph_scores(
            cache["score::Adam-aware 1st Order In-Run"],
            gradients, neighbors, weights, args.smoothing,
        )
        second, _, _ = conservative_gradient_graph_scores(
            cache["score::Adam-aware 2nd Order In-Run"],
            gradients, neighbors, weights, args.smoothing,
        )
        payload = {
            "fingerprint": cache["fingerprint"],
            "graph_neighbors": neighbors,
            "graph_weights": weights,
            "graph_gated_weights": gated_weights,
            "graph_confidence": confidence,
            "graph_metadata": json.dumps({
                "version": "graph_v3_conservative_diverse",
                "source_cache": str(args.cache),
                "smoothing": args.smoothing,
                "diversity_lambda_fraction": 0.5,
                "gradient_agreement_gate": True,
                "one_sided_unsupported_value_penalty": True,
                "historical_diffusion": False,
            }),
            "score::Graph Adam-aware 1st Order In-Run": first,
            "score::Graph Adam-aware 2nd Order In-Run": second,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
