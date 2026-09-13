"""Create the 100k main table with mean(std), ranks, and editable plot data."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

from pile_main_100k.common import EXPERIMENT, METHODS


METRICS = ["Noise AUROC", "Domain nRecall@1K", "20% Val. Accuracy (%)", "40% Val. Accuracy (%)", "60% Val. Accuracy (%)", "80% Val. Accuracy (%)"]


def read(path: Path):
    return json.loads(path.read_text())


def selection(budget: int):
    revised = EXPERIMENT / f"selection_{budget}" / "results_100k_graph_v2.json"
    merged = revised if revised.exists() else EXPERIMENT / f"selection_{budget}" / "results_100k.json"
    if merged.exists():
        values = read(merged)["results"]
        if set(values) != set(METHODS):
            missing = sorted(set(METHODS) - set(values))
            extra = sorted(set(values) - set(METHODS))
            raise RuntimeError(
                f"Incomplete selection results for {budget}% in {merged}: "
                f"missing={missing} extra={extra}"
            )
        return values

    values = {}
    for gpu in (0, 1):
        values.update(read(EXPERIMENT / f"selection_{budget}" / f"results_gpu{gpu}.json")["results"])
    if set(values) != set(METHODS):
        raise RuntimeError(f"Incomplete selection results for {budget}%")
    return values


def decorate(text: str, index: int, first: int, second: int):
    if index == first:
        return f"**{text}**"
    if index == second:
        return f"<u>{text}</u>"
    return text


def main() -> None:
    noise = read(EXPERIMENT / "noise_auroc" / "results_graph_v2.json")["details"]
    domain = read(EXPERIMENT / "domain_recall" / "results_graph_v2.json")["details"]
    selected = {budget: selection(budget) for budget in (20, 40, 60, 80)}
    sources = [noise, domain, selected[20], selected[40], selected[60], selected[80]]
    scales = np.asarray([1.0, 1.0, 100.0, 100.0, 100.0, 100.0])
    runs = np.asarray([[[source[method]["runs"][seed] for seed in range(3)] for source in sources] for method in METHODS])
    runs *= scales[None, :, None]
    means, stds = runs.mean(2), runs.std(2)
    run_ranks = np.empty_like(runs)
    for metric in range(len(METRICS)):
        for seed in range(3):
            run_ranks[:, metric, seed] = rankdata(-runs[:, metric, seed], method="average")
    average_rank_runs = run_ranks.mean(1)
    rank_means, rank_stds = average_rank_runs.mean(1), average_rank_runs.std(1)
    best = [int(np.argsort(-means[:, col], kind="stable")[0]) for col in range(6)]
    second = [int(np.argsort(-means[:, col], kind="stable")[1]) for col in range(6)]
    best.append(int(np.argsort(rank_means, kind="stable")[0]))
    second.append(int(np.argsort(rank_means, kind="stable")[1]))

    output = EXPERIMENT / "results"
    output.mkdir(parents=True, exist_ok=True)
    csv_rows = []
    for i, method in enumerate(METHODS):
        row = {"Method": method}
        for j, metric in enumerate(METRICS):
            row[f"{metric} mean"] = means[i, j]
            row[f"{metric} std"] = stds[i, j]
            for seed in range(3):
                row[f"{metric} run{seed}"] = runs[i, j, seed]
        row["Avg. Rank mean"], row["Avg. Rank std"] = rank_means[i], rank_stds[i]
        for seed in range(3):
            row[f"Avg. Rank run{seed}"] = average_rank_runs[i, seed]
        csv_rows.append(row)
    with (output / "main_table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0])); writer.writeheader(); writer.writerows(csv_rows)
    with (output / "plot_data.csv").open("w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["method", "metric", "seed", "value"])
        for i, method in enumerate(METHODS):
            for j, metric in enumerate(METRICS):
                for seed in range(3): writer.writerow([method, metric, seed, runs[i, j, seed]])

    header = "| Method | " + " | ".join(f"{metric} ↑" for metric in METRICS) + " | Avg. Rank ↓ |"
    lines = [header, "|---|" + "---:|" * 7]
    for i, method in enumerate(METHODS):
        cells = []
        for j in range(6):
            decimals = 4 if j < 2 else 3
            value = f"{means[i,j]:.{decimals}f} ({stds[i,j]:.3f})"
            cells.append(decorate(value, i, best[j], second[j]))
        cells.append(decorate(f"{rank_means[i]:.2f} ({rank_stds[i]:.3f})", i, best[6], second[6]))
        lines.append(f"| {method} | " + " | ".join(cells) + " |")

    # Pad the Markdown source so every column delimiter is vertically aligned.
    table_cells = [[cell.strip() for cell in line.strip("|").split("|")] for line in lines]
    widths = [max(len(row[col]) for row in table_cells) for col in range(len(table_cells[0]))]
    aligned_lines = []
    for row_index, row in enumerate(table_cells):
        if row_index == 1:
            padded = ["-" * widths[0], *["-" * (width - 1) + ":" for width in widths[1:]]]
        else:
            padded = [row[0].ljust(widths[0]), *[row[col].rjust(widths[col]) for col in range(1, len(row))]]
        aligned_lines.append("| " + " | ".join(padded) + " |")
    lines = aligned_lines
    lines.extend([
        "",
        "Note: selection accuracy is displayed as a percentage; raw JSON stores a 0-1 proportion.",
        "Standard deviations use three seeds and population std (ddof=0). Exact zeros are retained.",
    ])
    (output / "main_table.md").write_text("\n".join(lines) + "\n")
    (output / "main_table.json").write_text(json.dumps({
        "metrics": METRICS, "methods": METHODS, "runs": runs.tolist(), "means": means.tolist(),
        "stds": stds.tolist(), "average_rank_runs": average_rank_runs.tolist(),
        "average_rank_means": rank_means.tolist(), "average_rank_stds": rank_stds.tolist(),
        "best_method_per_column": [METHODS[i] for i in best], "second_method_per_column": [METHODS[i] for i in second],
        "display": {"selection_accuracy_unit": "percent", "raw_selection_unit": "proportion",
                    "std_ddof": 0, "decimals": {"auroc_recall_mean": 4, "selection_accuracy_mean": 3,
                                                   "standard_deviation": 3}},
        "graph_method_boundary": (
            "Graph selection rows use two-step full-vector semantic diffusion "
            "and symmetric-graph greedy diversity selection. Noise/domain provenance is recorded separately."
        ),
    }, indent=2) + "\n")

    rank_rows = []
    mean_metric_ranks = np.column_stack([rankdata(-means[:, col], method="average") for col in range(len(METRICS))])
    with (output / "per_metric_ranks.csv").open("w", newline="") as handle:
        fieldnames = ["Method", *METRICS, "Avg. Rank mean", "Avg. Rank std"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for i, method in enumerate(METHODS):
            row = {"Method": method, "Avg. Rank mean": rank_means[i], "Avg. Rank std": rank_stds[i]}
            row.update({metric: mean_metric_ranks[i, j] for j, metric in enumerate(METRICS)})
            rank_rows.append(row)
            writer.writerow(row)

    (output / "artifact_manifest.json").write_text(json.dumps({
        "table_source": "main_table.csv",
        "plot_source": "plot_data.csv",
        "rank_source": "per_metric_ranks.csv",
        "raw_sources": {
            "noise": "../noise_auroc/results_graph_v2.json with frozen non-Graph rows",
            "domain": "../domain_recall/results_graph_v2.json with frozen non-Graph rows",
            "selection": "../selection_{20,40,60,80}/results_100k_graph_v2.json; Graph cells from raw_100k_graph_v2",
            "dataset": "../../data/Pile_large/manifest.json"
        },
        "figure_formats": ["png", "pdf", "svg"],
        "display": {"selection_accuracy_unit": "percent", "std_ddof": 0, "decimals": {"auroc_recall_mean": 4, "selection_accuracy_mean": 3,
                                                   "standard_deviation": 3}},
        "graph_method_boundary": (
            "Selection uses Graph v2 trajectory replay and symmetric-graph diversity; "
            "accuracy is displayed in percent while raw JSON remains a 0-1 proportion."
        )
    }, indent=2) + "\n")

    normalized = np.empty_like(means)
    for col in range(6):
        normalized[:, col] = (means[:, col] - means[:, col].min()) / max(np.ptp(means[:, col]), 1e-12)
    fig, ax = plt.subplots(figsize=(12, 7))
    image = ax.imshow(normalized, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(6), [m.replace(" ", "\n", 1) for m in METRICS], fontsize=8)
    ax.set_yticks(range(len(METHODS)), METHODS, fontsize=8)
    for i in range(len(METHODS)):
        for j in range(6):
            decimals = 4 if j < 2 else 3
            label = f"{means[i,j]:.{decimals}f}\n({stds[i,j]:.{decimals}f})"
            ax.text(j, i, label, ha="center", va="center", fontsize=6,
                    fontweight="bold" if i == best[j] else "normal",
                    color="white" if normalized[i,j] < .55 else "black")
            if i == second[j]:
                ax.plot([j-.25, j+.25], [i+.28, i+.28], color="white" if normalized[i,j] < .55 else "black", linewidth=.8)
    fig.colorbar(image, ax=ax, label="Column-normalized performance")
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"main_table_heatmap.{suffix}", dpi=220 if suffix == "png" else None)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
