"""Run Table 4 mislabeled-data detection and write comparison tables."""

from __future__ import annotations

import csv
import importlib
import json
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np

from table4_common import EXPECTED_FINGERPRINT, ROOT, build_context, mislabeled_auroc


METHOD_MODULES = (
    "table4_cifar_mislabel_detection.retraining_data_shapley",
    "table4_cifar_mislabel_detection.empirical_influence",
    "table4_cifar_mislabel_detection.datamodels",
    "table4_cifar_mislabel_detection.knn_shapley",
    "table4_cifar_mislabel_detection.influence_function",
    "table4_cifar_mislabel_detection.less",
    "table4_cifar_mislabel_detection.first_order_inrun",
    "table4_cifar_mislabel_detection.second_order_inrun",
    "table4_cifar_mislabel_detection.trak",
    "table4_cifar_mislabel_detection.adam_inrun",
    "table4_cifar_mislabel_detection.graph_inrun",
)

PAPER_RESULTS = OrderedDict(
    [
        ("Retraining-based Data Shapley", (0.582, 0.029)),
        ("Empirical Influence Function", (0.552, 0.017)),
        ("Datamodels", (0.520, 0.008)),
        ("KNN-Shapley", (0.760, 0.018)),
        ("Influence function", (0.654, 0.054)),
        ("LESS", (0.550, 0.032)),
        ("1st Order In-Run Data Shapley", (0.678, 0.045)),
        ("2nd Order In-Run Data Shapley", (0.680, 0.048)),
        ("TRAK (1 cpt)", (0.511, 0.012)),
        ("TRAK (5 cpts)", (0.542, 0.025)),
        ("TRAK (15 cpts)", (0.609, 0.028)),
        ("TRAK (25 cpts)", (0.617, 0.021)),
    ]
)


def summarize(records):
    summary = OrderedDict()
    for method, runs in records.items():
        values = np.asarray(runs, dtype=np.float64)
        summary[method] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "runs": values.tolist(),
        }
    return summary


def format_result_table(summary):
    rows = [["Method", "AUROC"]]
    rows.extend(
        [method, f'{stats["mean"]:.3f} ({stats["std"]:.3f})']
        for method, stats in summary.items()
    )
    widths = [max(len(row[column]) for row in rows) for column in range(2)]
    return "\n".join(
        "  ".join(cell.ljust(widths[column]) for column, cell in enumerate(row))
        for row in rows
    )


def write_outputs(summary, run_metadata):
    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": {
            "dataset": "CIFAR10",
            "subset_size": 1000,
            "mislabel_fraction": 0.10,
            "runs": 3,
            "seeds": [0, 1, 2],
            "model": "ImageNet-pretrained ResNet18",
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "label_smoothing": 0.1,
            "epochs": 50,
            "batch_size": 64,
            "lr_schedule": "multiply by 0.1 every 10 epochs",
            "positive_class": "mislabeled",
            "anomaly_score": "negative data-attribution score",
            "score_cache_fingerprint": EXPECTED_FINGERPRINT,
            "implementation_note": (
                "Uses the exact completed Table 5 score vectors. TRAK, LESS, influence, and "
                "In-Run methods retain the approximation boundaries documented there."
            ),
        },
        "runs": run_metadata,
        "paper_reference": {
            name: {"mean": mean, "std": std} for name, (mean, std) in PAPER_RESULTS.items()
        },
        "summary": summary,
    }
    (output_dir / "table4_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    with (output_dir / "table4_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "mean_auroc", "std_auroc", "run_0", "run_1", "run_2", "paper_mean", "paper_std"])
        for method, stats in summary.items():
            paper = PAPER_RESULTS.get(method, ("", ""))
            writer.writerow([method, stats["mean"], stats["std"], *stats["runs"], *paper])

    result_table = format_result_table(summary)
    lines = [
        "# CIFAR10 Table 4 Results",
        "",
        "```text",
        result_table,
        "```",
        "",
        "## Paper Reference Comparison",
        "",
        "| Method | Reproduced | Paper | Delta |",
        "|---|---:|---:|---:|",
    ]
    for method, (paper_mean, paper_std) in PAPER_RESULTS.items():
        stats = summary[method]
        reproduced = f'{stats["mean"]:.3f} ({stats["std"]:.3f})'
        paper = f"{paper_mean:.3f} ({paper_std:.3f})"
        delta = f'{stats["mean"] - paper_mean:+.3f}'
        lines.append(f"| {method} | {reproduced} | {paper} | {delta} |")
    lines.extend(
        [
            "",
            "Standard deviations use the population convention over three runs.",
        ]
    )
    (output_dir / "table4_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "table4_results.txt").write_text(result_table + "\n", encoding="utf-8")
    return output_dir


def main():
    os.chdir(ROOT)
    modules = [importlib.import_module(name) for name in METHOD_MODULES]
    records = OrderedDict()
    run_metadata = []
    for run_index in range(3):
        context = build_context(run_index)
        run_metadata.append(
            {
                "run": run_index,
                "seed": context.seed,
                "points": len(context.train_dataset),
                "mislabeled": int(context.corrupted_mask.sum()),
            }
        )
        print(f"Run {run_index + 1}/3: n=1000, mislabeled=100", flush=True)
        seen = set()
        for module in modules:
            for method, scores in module.score(context).items():
                if method in seen:
                    raise RuntimeError(f"Duplicate method in run {run_index}: {method}")
                seen.add(method)
                auroc = mislabeled_auroc(context.corrupted_mask, scores)
                records.setdefault(method, []).append(auroc)
                print(f"  {method:42s} AUROC={auroc:.4f}", flush=True)
        if len(seen) != 16:
            raise RuntimeError(f"Expected 16 methods, got {len(seen)}")

    if any(len(values) != 3 for values in records.values()):
        raise RuntimeError("Every method must contain exactly three AUROC values")
    summary = summarize(records)
    output_dir = write_outputs(summary, run_metadata)
    print("\n" + format_result_table(summary), flush=True)
    print(f"\nSaved to {output_dir}", flush=True)


if __name__ == "__main__":
    main()

