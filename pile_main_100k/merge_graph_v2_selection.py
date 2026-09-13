"""Overlay Graph v2 cells on the frozen 100k baseline selection results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pile_main_100k.common import METHODS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, required=True)
    args = parser.parse_args()
    folder = Path(f"pile_main_100k/selection_{args.budget}")
    base_path = folder / "results_100k.json"
    payload = json.loads(base_path.read_text())
    sources = list(payload["raw_sources"])
    for method_index in (10, 11):
        method = METHODS[method_index]
        cells = []
        for seed in range(3):
            path = folder / "raw_100k_graph_v2" / f"method{method_index}_seed{seed}.json"
            cell = json.loads(path.read_text())
            if (
                cell["method"] != method
                or cell["seed"] != seed
                or cell["budget"] != args.budget
                or cell["selection"]["strategy"] != "symmetric_graph_greedy"
            ):
                raise RuntimeError(f"Mismatched Graph v2 cell: {path}")
            cells.append(cell)
            sources.append(str(path))
        accuracy = [float(cell["accuracy"]) for cell in cells]
        losses = [float(cell["val_loss"]) for cell in cells]
        payload["results"][method] = {
            "runs": accuracy,
            "mean": float(np.mean(accuracy)),
            "std": float(np.std(accuracy)),
            "val_loss_runs": losses,
            "val_loss_mean": float(np.mean(losses)),
            "val_loss_std": float(np.std(losses)),
            "selection_diagnostics": [cell["selection"] for cell in cells],
            "selected_indices_sha256": [cell["selected_indices_sha256"] for cell in cells],
        }
    payload["raw_sources"] = sources
    payload["graph_version"] = {
        "name": "graph_v2_dynamic_diverse",
        "alpha": 0.2,
        "lambda_fraction": 0.05,
        "base_results": str(base_path),
    }
    output = folder / "results_100k_graph_v2.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"merged={output}")


if __name__ == "__main__":
    main()
