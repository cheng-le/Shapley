"""Merge all experiment groups into the paper main table and visualization."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

from pile_main_100k.common import EXPERIMENT, METHODS


def read_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required result: {path}")
    return json.loads(path.read_text())


def merge_selection(budget: int):
    folder = EXPERIMENT / f"selection_{budget}"
    merged = {}
    for part in (folder / "results_gpu0.json", folder / "results_gpu1.json"):
        merged.update(read_json(part)["results"])
    missing = set(METHODS) - set(merged)
    if missing:
        raise RuntimeError(f"Selection {budget} is missing methods: {sorted(missing)}")
    return merged


def main() -> None:
    noise = read_json(EXPERIMENT / "noise_auroc" / "results.json")["results"]
    domain = read_json(EXPERIMENT / "domain_recall" / "results.json")["results"]
    selections = {budget: merge_selection(budget) for budget in (20, 40, 60, 80)}
    columns = ["Noise AUROC", "Domain nRecall@1K", "20% Val. Loss", "40% Val. Loss", "60% Val. Loss", "80% Val. Loss"]
    matrix = np.asarray([[noise[m], domain[m], *[selections[b][m]["mean"] for b in (20, 40, 60, 80)]] for m in METHODS])
    rank_matrix = np.column_stack([
        rankdata(-matrix[:, 0], method="average"), rankdata(-matrix[:, 1], method="average"),
        *[rankdata(matrix[:, index], method="average") for index in range(2, 6)],
    ])
    average_rank = rank_matrix.mean(1)
    output = EXPERIMENT / "results"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, method in enumerate(METHODS):
        rows.append({"Method": method, **{column: float(matrix[index, j]) for j, column in enumerate(columns)}, "Avg. Rank": float(average_rank[index])})
    with (output / "main_table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (output / "main_table.json").write_text(json.dumps({"rows": rows, "per_metric_ranks": rank_matrix.tolist()}, indent=2) + "\n")
    header = "| Method | Noise AUROC ↑ | Domain nRecall@1K ↑ | 20% Val. Loss ↓ | 40% Val. Loss ↓ | 60% Val. Loss ↓ | 80% Val. Loss ↓ | Avg. Rank ↓ |"
    lines = [header, "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['Method']} | " + " | ".join(f"{row[column]:.3f}" for column in columns) + f" | {row['Avg. Rank']:.2f} |")
    (output / "main_table.md").write_text("\n".join(lines) + "\n")

    normalized = np.empty_like(matrix)
    for col in range(matrix.shape[1]):
        values = matrix[:, col]
        normalized[:, col] = (values - values.min()) / max(values.max() - values.min(), 1e-12)
        if col >= 2:
            normalized[:, col] = 1.0 - normalized[:, col]
    fig, ax = plt.subplots(figsize=(10, 6.5))
    image = ax.imshow(normalized, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(6), ["Noise\nAUROC", "Domain\nnRecall@1K", "20%\nVal. Loss", "40%\nVal. Loss", "60%\nVal. Loss", "80%\nVal. Loss"])
    ax.set_yticks(range(len(METHODS)), METHODS, fontsize=8)
    for i in range(len(METHODS)):
        for j in range(6):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", color="white" if normalized[i, j] < .55 else "black", fontsize=7)
    fig.colorbar(image, ax=ax, label="Column-normalized performance (higher is better)")
    fig.tight_layout()
    fig.savefig(output / "main_table_heatmap.png", dpi=220)
    fig.savefig(output / "main_table_heatmap.pdf")
    fig.savefig(output / "main_table_heatmap.svg")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
