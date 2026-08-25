"""Run all Table 5 baselines and emit mean/std test-accuracy tables."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
from pathlib import Path

import numpy as np

from table5_common import (
    Table5Config,
    add_shared_cli,
    build_run_context,
    config_fingerprint,
    config_from_args,
    load_or_compute_scores,
    train_selected_and_evaluate,
)


METHOD_MODULES = [
    "random_baseline",
    "influence_function",
    "trak",
    "less",
    "data_shapley",
    "datamodels",
    "empirical_influence",
    "knn_shapley",
    "first_order_inrun",
    "second_order_inrun",
    "adam_inrun",
    "graph_inrun",
]


def _load_selection_cache(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return {key: float(value) for key, value in json.load(handle).items()}


def _save_selection_cache(path: Path, values: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(values, handle, indent=2, sort_keys=True)
    temporary.replace(path)


def _format_table(summary: dict, budgets: tuple[float, ...]) -> str:
    headers = ["Method"] + [f"{int(budget * 100)}%" for budget in budgets]
    rows = [headers]
    for method, values in summary.items():
        rows.append(
            [method]
            + [f"{values[str(budget)]['mean']:.3f} ({values[str(budget)]['std']:.3f})" for budget in budgets]
        )
    widths = [max(len(row[col]) for row in rows) for col in range(len(headers))]
    return "\n".join("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) for row in rows)


def _write_outputs(config: Table5Config, records: dict[str, dict[str, list[float]]]) -> dict:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for method, by_budget in records.items():
        summary[method] = {}
        for budget, accuracies in by_budget.items():
            values = np.asarray(accuracies, dtype=np.float64)
            summary[method][budget] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "runs": values.tolist(),
            }

    payload = {
        "protocol": {
            **config.__dict__,
            "budgets": list(config.budgets),
            "implementation_note": (
                "TRAK, LESS, influence function, and In-Run methods use last-layer gradient "
                "representations; see README.md for exact-vs-approximate method boundaries."
            ),
        },
        "summary": summary,
    }
    with (output_dir / "table5_results.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    with (output_dir / "table5_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "budget", "mean", "std"] + [f"run_{i}" for i in range(config.runs)])
        for method, by_budget in summary.items():
            for budget, stats in by_budget.items():
                writer.writerow([method, budget, stats["mean"], stats["std"], *stats["runs"]])

    markdown = _format_table(summary, config.budgets)
    md_lines = ["# CIFAR10 Table 5 Results", "", "```text", markdown, "```", ""]
    (output_dir / "table5_results.md").write_text("\n".join(md_lines), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce Table 5 data-selection comparisons on local CIFAR10.")
    add_shared_cli(parser)
    parser.add_argument("--budgets", default="0.2,0.4,0.6,0.8")
    parser.add_argument("--methods", default=",".join(METHOD_MODULES), help="Comma-separated module names.")
    parser.add_argument("--influence-cg-steps", type=int, default=50)
    parser.add_argument("--influence-damping", type=float, default=1e-2)
    parser.add_argument("--trak-ridge", type=float, default=1e-3)
    parser.add_argument("--datamodel-ridge", type=float, default=1e-2)
    parser.add_argument("--knn-k", type=int, default=5)
    args = parser.parse_args()
    config = config_from_args(args)
    config.budgets = tuple(float(value) for value in args.budgets.split(","))
    modules = [importlib.import_module(name.strip()) for name in args.methods.split(",") if name.strip()]

    fingerprint = config_fingerprint(config)
    cache_path = Path(config.output_dir) / f"selection_accuracies_{fingerprint}.json"
    selection_cache = _load_selection_cache(cache_path)
    records: dict[str, dict[str, list[float]]] = {}

    for run_index in range(config.runs):
        context = build_run_context(config, run_index)
        print(
            f"\nRun {run_index + 1}/{config.runs}: n={len(context.train_dataset)}, "
            f"mislabeled={int(context.corrupted_mask.sum())}",
            flush=True,
        )
        for module in modules:
            cache_version = getattr(module, "CACHE_VERSION", "")
            score_cache_key = module.METHOD_NAME
            if cache_version:
                score_cache_key = f"{score_cache_key} {cache_version}"
            variants = load_or_compute_scores(context, score_cache_key, module.score)
            for method_name, scores in variants.items():
                if len(scores) != len(context.train_dataset):
                    raise ValueError(f"{method_name} returned {len(scores)} scores for {len(context.train_dataset)} points")
                records.setdefault(method_name, {str(budget): [] for budget in config.budgets})
                ranking = np.argsort(np.nan_to_num(scores, nan=-np.inf))[::-1]
                for budget_index, budget in enumerate(config.budgets):
                    count = max(1, int(round(len(ranking) * budget)))
                    key = f"{method_name}|run={run_index}|budget={budget}|{fingerprint}"
                    if cache_version:
                        key = f"{key}|method_version={cache_version}"
                    if key not in selection_cache or config.force:
                        selection_cache[key] = train_selected_and_evaluate(
                            context,
                            ranking[:count],
                            seed_offset=100_000 + budget_index,
                        )
                        _save_selection_cache(cache_path, selection_cache)
                    accuracy = selection_cache[key]
                    records[method_name][str(budget)].append(accuracy)
                    print(f"  {method_name:38s} {budget:>4.0%}: {accuracy:.4f}", flush=True)
        summary = _write_outputs(config, records)

    print("\n" + _format_table(summary, config.budgets))
    print(f"\nSaved to {Path(config.output_dir).resolve()}")


if __name__ == "__main__":
    main()

