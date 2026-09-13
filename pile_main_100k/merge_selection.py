"""Merge process-isolated selection cells into the two expected GPU shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    args = parser.parse_args()
    results = {}
    sources = []
    for method_index, method in enumerate(args.methods):
        runs = []
        for seed in (0, 1, 2):
            path = args.raw_dir / f"method{method_index}_seed{seed}.json"
            payload = json.loads(path.read_text())
            runs.append(float(payload["results"][method]["runs"][0]))
            sources.append(str(path))
        results[method] = {"runs": runs, "mean": float(np.mean(runs)), "std": float(np.std(runs))}
    payload = {"budget": args.budget, "selected_corpora": args.budget * 100, "epochs": 1, "results": results, "raw_sources": sources}
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"merged={args.output} cells={len(sources)}")


if __name__ == "__main__":
    main()
