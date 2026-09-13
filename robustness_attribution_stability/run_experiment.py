"""Analyze robustness and cross-seed stability from frozen Pile/CIFAR caches."""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results"
FINGERPRINT = "16a6f42d74e8"
PAIRS = (
    ("Adam-aware 1st Order In-Run", "Graph Adam-aware 1st Order In-Run"),
    ("Adam-aware 2nd Order In-Run", "Graph Adam-aware 2nd Order In-Run"),
)
PRIMARY_METHODS = tuple(method for pair in PAIRS for method in pair)
PILE_FRACTIONS = (0.01, 0.05, 0.10, 0.20)
CIFAR_FRACTIONS = (0.10, 0.20, 0.40)


def finite(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite one-dimensional array")
    return array


def contamination(scores: np.ndarray, corrupted: np.ndarray, fraction: float) -> float:
    count = max(1, int(round(len(scores) * fraction)))
    selected = np.argpartition(scores, len(scores) - count)[-count:]
    return float(np.mean(corrupted[selected]))


def standardized(values: np.ndarray) -> np.ndarray:
    scale = float(np.std(values))
    return (values - float(np.mean(values))) / max(scale, 1e-12)


def correlation_value(result) -> float:
    value = getattr(result, "statistic", getattr(result, "correlation", result[0]))
    return float(value)


def stability(scores: list[np.ndarray]) -> dict:
    pair_rows = []
    for left, right in combinations(range(len(scores)), 2):
        row = {
            "seed_pair": [left, right],
            "spearman": correlation_value(spearmanr(scores[left], scores[right])),
            "kendall_tau": correlation_value(kendalltau(scores[left], scores[right])),
        }
        for fraction in PILE_FRACTIONS:
            count = max(1, int(round(len(scores[left]) * fraction)))
            left_top = set(np.argpartition(scores[left], len(scores[left]) - count)[-count:])
            right_top = set(np.argpartition(scores[right], len(scores[right]) - count)[-count:])
            row[f"top_{int(fraction * 100)}pct_overlap"] = len(left_top & right_top) / count
        pair_rows.append(row)
    stacked = np.stack([standardized(values) for values in scores])
    return {
        "pairs": pair_rows,
        "mean_spearman": float(np.mean([row["spearman"] for row in pair_rows])),
        "mean_kendall_tau": float(np.mean([row["kendall_tau"] for row in pair_rows])),
        "mean_standardized_dispersion": float(np.mean(np.std(stacked, axis=0, ddof=0))),
        **{
            f"mean_top_{int(fraction * 100)}pct_overlap": float(
                np.mean([row[f"top_{int(fraction * 100)}pct_overlap"] for row in pair_rows])
            )
            for fraction in PILE_FRACTIONS
        },
    }


def add_metric(rows: list[dict], dataset: str, seed: int, method: str, metric: str, value: float) -> None:
    rows.append({"dataset": dataset, "seed": seed, "method": method, "metric": metric, "value": float(value)})


def analyze_pile(metric_rows: list[dict], change_rows: list[dict]) -> tuple[dict, dict]:
    cache_dir = ROOT / "pile_main_100k" / "cache"
    graph_dir = ROOT / "pile_main_100k" / "cache_graph_v2"
    clean_scores = {method: [] for method in PRIMARY_METHODS}
    confidence_bins: dict[str, dict[str, list[dict]]] = {graph: {} for _, graph in PAIRS}

    for seed in range(3):
        with np.load(cache_dir / f"noise_seed{seed}.npz", allow_pickle=False) as raw, np.load(
            graph_dir / f"noise_seed{seed}_graph_v2.npz", allow_pickle=False
        ) as graph:
            corrupted = np.asarray(raw["noise_labels"], dtype=bool)
            if corrupted.sum() != 10_000:
                raise ValueError(f"Pile seed {seed} has {corrupted.sum()} corrupted rows; expected 10000")
            methods = sorted(key[len("score::"):] for key in raw.files if key.startswith("score::"))
            for method in methods:
                source = graph if method.startswith("Graph Adam-aware") else raw
                scores = finite(source[f"score::{method}"], f"Pile {method} seed {seed}")
                add_metric(metric_rows, "Pile", seed, method, "corruption_auroc", roc_auc_score(corrupted, -scores))
                for fraction in PILE_FRACTIONS:
                    add_metric(metric_rows, "Pile", seed, method, f"top_{int(fraction * 100)}pct_contamination", contamination(scores, corrupted, fraction))

            confidence = finite(graph["graph_confidence"], f"Pile confidence seed {seed}")
            boundaries = np.quantile(confidence, [0.25, 0.50, 0.75])
            bins = np.digitize(confidence, boundaries, right=True)
            for raw_method, graph_method in PAIRS:
                raw_scores = finite(raw[f"score::{raw_method}"], f"Pile {raw_method}")
                graph_scores = finite(graph[f"score::{graph_method}"], f"Pile {graph_method}")
                raw_z, graph_z = standardized(raw_scores), standardized(graph_scores)
                delta_z = graph_z - raw_z
                for index in np.flatnonzero(corrupted):
                    change_rows.append({
                        "dataset": "Pile", "seed": seed, "index": int(index), "order": raw_method.split()[1],
                        "corrupted": 1, "raw_score": raw_scores[index], "graph_score": graph_scores[index],
                        "standardized_delta": delta_z[index], "graph_confidence": confidence[index],
                    })
                bin_rows = []
                for bin_index in range(4):
                    mask = bins == bin_index
                    noisy = mask & corrupted
                    bin_rows.append({
                        "seed": seed,
                        "quartile": bin_index + 1,
                        "count": int(mask.sum()),
                        "confidence_min": float(confidence[mask].min()),
                        "confidence_max": float(confidence[mask].max()),
                        "corruption_rate": float(corrupted[mask].mean()),
                        "corrupted_mean_standardized_delta": float(delta_z[noisy].mean()),
                        "corrupted_promotion_rate": float(np.mean(delta_z[noisy] > 0.0)),
                    })
                confidence_bins[graph_method][str(seed)] = bin_rows

        with np.load(cache_dir / f"clean_seed{seed}.npz", allow_pickle=False) as raw, np.load(
            graph_dir / f"clean_seed{seed}_graph_v2.npz", allow_pickle=False
        ) as graph:
            for raw_method, graph_method in PAIRS:
                clean_scores[raw_method].append(finite(raw[f"score::{raw_method}"], raw_method))
                clean_scores[graph_method].append(finite(graph[f"score::{graph_method}"], graph_method))

    return {method: stability(values) for method, values in clean_scores.items()}, confidence_bins


def load_score_cache(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    # NumPy 2 renamed numpy.core to numpy._core. Register read-only aliases so
    # caches produced by NumPy 2 can be loaded in the project's NumPy 1.x env.
    import sys

    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return {name: finite(values, f"{path.name}:{name}") for name, values in payload["scores"].items()}


def cifar_corruption_mask(seed: int, size: int = 1_000) -> np.ndarray:
    rng = np.random.default_rng(seed + 10_000)
    mask = np.zeros(size, dtype=bool)
    mask[rng.choice(size, size=100, replace=False)] = True
    return mask


def analyze_cifar(metric_rows: list[dict], change_rows: list[dict]) -> None:
    score_dir = ROOT / "table5_cifar_comparison" / "results" / "scores"
    reference = json.loads((ROOT / "table4_cifar_mislabel_detection" / "results" / "table4_results.json").read_text())
    for seed in range(3):
        raw = load_score_cache(score_dir / f"adam-aware_in-run_run{seed}_{FINGERPRINT}.pt")
        graph = load_score_cache(score_dir / f"graph_adam-aware_in-run_semantic-conservative-v3_run{seed}_{FINGERPRINT}.pt")
        corrupted = cifar_corruption_mask(seed)
        for raw_method, graph_method in PAIRS:
            for method, scores in ((raw_method, raw[raw_method]), (graph_method, graph[graph_method])):
                auroc = float(roc_auc_score(corrupted, -scores))
                expected = float(reference["summary"][method]["runs"][seed])
                if not np.isclose(auroc, expected, atol=1e-12):
                    raise ValueError(f"CIFAR cache mismatch for {method}, seed {seed}: {auroc} != {expected}")
                add_metric(metric_rows, "CIFAR-10", seed, method, "corruption_auroc", auroc)
                for fraction in CIFAR_FRACTIONS:
                    add_metric(metric_rows, "CIFAR-10", seed, method, f"top_{int(fraction * 100)}pct_contamination", contamination(scores, corrupted, fraction))
            raw_z, graph_z = standardized(raw[raw_method]), standardized(graph[graph_method])
            for index in np.flatnonzero(corrupted):
                change_rows.append({
                    "dataset": "CIFAR-10", "seed": seed, "index": int(index), "order": raw_method.split()[1],
                    "corrupted": 1, "raw_score": raw[raw_method][index], "graph_score": graph[graph_method][index],
                    "standardized_delta": graph_z[index] - raw_z[index], "graph_confidence": "",
                })


def summarize_metrics(rows: list[dict]) -> dict:
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        grouped.setdefault((row["dataset"], row["method"], row["metric"]), []).append(row["value"])
    output: dict[str, dict] = {}
    for (dataset, method, metric), values in grouped.items():
        array = np.asarray(values)
        output.setdefault(dataset, {}).setdefault(method, {})[metric] = {
            "runs": array.tolist(), "mean": float(array.mean()), "std": float(array.std(ddof=0))
        }
    return output


def transition_summary(change_rows: list[dict]) -> dict:
    output = {}
    for dataset in ("Pile", "CIFAR-10"):
        for order in ("1st", "2nd"):
            values = np.asarray([row["standardized_delta"] for row in change_rows if row["dataset"] == dataset and row["order"] == order])
            output.setdefault(dataset, {})[order] = {
                "corrupted_count": int(len(values)),
                "mean_standardized_delta": float(values.mean()),
                "promotion_rate": float(np.mean(values > 0.0)),
            }
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(stats: dict) -> str:
    return f'{stats["mean"]:.3f} ({stats["std"]:.3f})'


def write_markdown(output: Path, summary: dict, pile_stability: dict, transitions: dict) -> None:
    lines = [
        "# Robustness and Attribution Stability Results", "",
        "## Corruption Robustness", "",
        "| Method | Pile AUROC | Pile Top-10% contamination | CIFAR-10 AUROC | CIFAR-10 Top-10% contamination |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in PRIMARY_METHODS:
        lines.append(
            f'| {method} | {fmt(summary["Pile"][method]["corruption_auroc"])} | '
            f'{fmt(summary["Pile"][method]["top_10pct_contamination"])} | '
            f'{fmt(summary["CIFAR-10"][method]["corruption_auroc"])} | '
            f'{fmt(summary["CIFAR-10"][method]["top_10pct_contamination"])} |'
        )
    lines.extend(["", "## Pile Cross-Seed Stability", "", "| Method | Spearman | Kendall tau | Top-10% overlap | Standardized dispersion |", "|---|---:|---:|---:|---:|"])
    for method in PRIMARY_METHODS:
        stats = pile_stability[method]
        lines.append(f'| {method} | {stats["mean_spearman"]:.3f} | {stats["mean_kendall_tau"]:.3f} | {stats["mean_top_10pct_overlap"]:.3f} | {stats["mean_standardized_dispersion"]:.3f} |')
    lines.extend(["", "## Corrupted-Node Graph Change", "", "| Dataset | Order | Standardized score delta | Promotion rate |", "|---|---:|---:|---:|"])
    for dataset in ("Pile", "CIFAR-10"):
        for order in ("1st", "2nd"):
            stats = transitions[dataset][order]
            lines.append(f'| {dataset} | {order} | {stats["mean_standardized_delta"]:.3f} | {stats["promotion_rate"]:.3f} |')
    lines.extend(["", "Lower contamination, dispersion, corrupted-node delta, and promotion rate are better; higher AUROC and rank correlations are better.", "Pile Graph v2 is analyzed as the frozen approximation documented by the main experiment, not as a strict online implementation.", ""])
    (output / "summary.md").write_text("\n".join(lines))


def plot_results(output: Path, summary: dict, pile_stability: dict) -> None:
    labels = ["Adam 1st", "Graph 1st", "Adam 2nd", "Graph 2nd"]
    x = np.arange(len(PRIMARY_METHODS))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    colors = ["#6b7280", "#14866d", "#9a6a38", "#2563a6"]
    for axis, dataset in zip(axes[:2], ("Pile", "CIFAR-10")):
        means = [summary[dataset][method]["corruption_auroc"]["mean"] for method in PRIMARY_METHODS]
        stds = [summary[dataset][method]["corruption_auroc"]["std"] for method in PRIMARY_METHODS]
        axis.bar(x, means, yerr=stds, color=colors, capsize=3)
        axis.axhline(0.5, color="#333333", linewidth=1, linestyle="--")
        axis.set_title(f"{dataset} corruption AUROC")
        axis.set_ylim(0.4, max(0.7, max(np.add(means, stds)) + 0.04))
        axis.set_xticks(x, labels, rotation=25, ha="right")
    axes[0].set_ylabel("AUROC")
    spearman = [pile_stability[method]["mean_spearman"] for method in PRIMARY_METHODS]
    axes[2].bar(x, spearman, color=colors)
    axes[2].set_title("Pile cross-seed rank stability")
    axes[2].set_ylabel("Mean Spearman correlation")
    axes[2].set_xticks(x, labels, rotation=25, ha="right")
    axes[2].set_ylim(min(0.0, min(spearman) - 0.05), 1.0)
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"robustness_stability.{suffix}", dpi=220 if suffix == "png" else None)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows, change_rows = [], []
    pile_stability, confidence_bins = analyze_pile(metric_rows, change_rows)
    analyze_cifar(metric_rows, change_rows)
    summary = summarize_metrics(metric_rows)
    transitions = transition_summary(change_rows)
    payload = {
        "protocol": {
            "seeds": [0, 1, 2], "std_ddof": 0, "pile_corruption_fraction": 0.10,
            "cifar_mislabel_fraction": 0.10, "pile_graph_version": "graph_v2_dynamic_diverse",
            "cifar_graph_version": "semantic-conservative-v3", "source_artifacts_are_read_only": True,
        },
        "metric_summary": summary,
        "pile_score_stability": pile_stability,
        "pile_confidence_bins": confidence_bins,
        "corrupted_node_transitions": transitions,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    write_csv(args.output_dir / "metrics.csv", metric_rows)
    write_csv(args.output_dir / "corrupted_score_changes.csv", change_rows)
    write_markdown(args.output_dir, summary, pile_stability, transitions)
    plot_results(args.output_dir, summary, pile_stability)
    print((args.output_dir / "summary.md").read_text())


if __name__ == "__main__":
    main()
