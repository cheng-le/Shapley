"""Derive trajectory-aware Graph scores from an existing 100k cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pile_main_100k.common import confidence_gated_graph_weights, replay_graph_trajectory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cache = np.load(args.cache, allow_pickle=False)
    neighbors, semantic_weights = cache["graph_neighbors"], cache["graph_weights"]
    gated_weights, confidence = confidence_gated_graph_weights(
        cache["gradients"], neighbors, semantic_weights
    )
    first = replay_graph_trajectory(
        cache["score::Adam-aware 1st Order In-Run"], neighbors, gated_weights,
        confidence, args.seed, alpha=args.alpha,
    )
    second = replay_graph_trajectory(
        cache["score::Adam-aware 2nd Order In-Run"], neighbors, gated_weights,
        confidence, args.seed, alpha=args.alpha, include_pair=True,
    )
    metadata = {
        "version": "graph_v2_dynamic_diverse",
        "source_cache": str(args.cache),
        "seed": args.seed,
        "alpha": args.alpha,
        "diversity_lambda_fraction": 0.05,
        "gradient_agreement_gate": True,
        "historical_diffusion": True,
        "in_batch_pair_surrogate_second_order": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        fingerprint=cache["fingerprint"],
        graph_neighbors=neighbors,
        graph_weights=semantic_weights,
        graph_gated_weights=gated_weights,
        graph_confidence=confidence,
        graph_metadata=json.dumps(metadata),
        **{
            "score::Graph Adam-aware 1st Order In-Run": first,
            "score::Graph Adam-aware 2nd Order In-Run": second,
        },
    )
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
