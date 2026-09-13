"""Aggregate three domain-retrieval runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pile_main_100k.common import METHODS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payloads = [json.loads(path.read_text()) for path in args.inputs]
    details = {}
    for method in METHODS:
        runs = [float(payload["results"][method]) for payload in payloads]
        details[method] = {"runs": runs, "mean": float(np.mean(runs)), "std": float(np.std(runs))}
    output = {
        "metric": payloads[0]["metric"], "definition": payloads[0]["definition"],
        "eligibility": payloads[0]["eligibility"], "eligible_domains": payloads[0]["eligible_domains"],
        "num_runs": len(payloads), "results": {m: v["mean"] for m, v in details.items()},
        "details": details, "run_sources": [str(path) for path in args.inputs],
        "per_run_per_domain": [payload["per_domain"] for payload in payloads],
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
