"""Compare frozen Graph selections with the corresponding In-Run baselines."""
import csv
import json
from pathlib import Path

import numpy as np

from diverse_data_selection.run_experiment import PILE, ROOT, digest, index_digest
from pile_main_100k.common import symmetric_graph

BUDGETS = (1, 2, 5, 10)
# Positive directions for coverage/entropy describe diversity, not model utility.
METRICS = {
    "val_loss": ("Validation loss", -1),
    "accuracy": ("Token accuracy (%)", 1),
    "affinity_per_item": ("Graph affinity per item", -1),
    "graph_coverage": ("One-hop graph coverage", 1),
    "domain_entropy": ("Domain entropy (nats)", 1),
    "domain_count": ("Represented domains", 1),
}


def main():
    folder = ROOT / "diverse_data_selection/results"
    out = folder / "inrun_comparison"
    out.mkdir(exist_ok=True)
    existing = list(csv.DictReader((folder / "metrics.csv").open()))
    sources = {folder / "metrics.csv", Path(__file__), PILE / "common.py"}
    fingerprint = json.loads((PILE / "token_cache/metadata.json").read_text())["fingerprint"]
    rows = []
    for seed in range(3):
        cache_path = PILE / f"cache/clean_seed{seed}.npz"
        sources.add(cache_path)
        with np.load(cache_path, allow_pickle=False) as cache:
            if str(cache["fingerprint"]) != fingerprint:
                raise ValueError("Cache fingerprint mismatch")
            adjacency = symmetric_graph(cache["graph_neighbors"], cache["graph_weights"])
            domains = cache["domains"]
            for order in (1, 2):
                method = f"{order}{'st' if order == 1 else 'nd'} Order In-Run Data Shapley"
                scores = cache[f"score::{method}"]
                for budget in BUDGETS:
                    count = round(len(scores) * budget / 100)
                    selected = np.argpartition(scores, -count)[-count:]
                    path = PILE / f"low_ratio_validation_loss/raw/method{order + 5}_seed{seed}_budget{budget}.json"
                    sources.add(path)
                    cell = json.loads(path.read_text())
                    expected = {"method": method, "seed": seed, "budget": budget,
                                "epochs": 1, "eval_part": "evaluation", "lr_schedule": "constant",
                                "protocol_split": "pile_main_100k/protocol_v2_split.json",
                                "selected_corpora": count, "selected_indices_sha256": index_digest(selected)}
                    if any(cell.get(key) != value for key, value in expected.items()) or cell["selection"]["strategy"] != "top_k":
                        raise ValueError(f"Baseline selection or protocol mismatch: {path}")
                    mask = np.zeros(len(scores), dtype=bool)
                    mask[selected] = True
                    covered = mask.copy()
                    covered[adjacency[mask].indices] = True
                    _, counts = np.unique(domains[selected], return_counts=True)
                    probabilities = counts / count
                    baseline = {"val_loss": cell["val_loss"], "accuracy": cell["accuracy"] * 100,
                                "affinity_per_item": float(adjacency[mask][:, mask].sum()) / (2 * count),
                                "graph_coverage": float(covered.mean()),
                                "domain_entropy": float(-np.sum(probabilities * np.log(probabilities))),
                                "domain_count": len(counts)}
                    matches = [r for r in existing if (int(r["seed"]), int(r["order"]), int(r["budget"]), r["arm"]) ==
                               (seed, order, budget, "Graph + diversity")]
                    if len(matches) != 1:
                        raise ValueError("Missing or duplicate Graph cell")
                    graph = matches[0]
                    sources.add(ROOT / graph["source"])
                    for metric in METRICS:
                        value = float(graph[metric]) * (100 if metric == "accuracy" else 1)
                        rows.append({"seed": seed, "order": order, "budget": budget, "metric": metric,
                                     "inrun": baseline[metric], "graph": value, "delta": value - baseline[metric]})
    groups = []
    for metric, (_, direction) in METRICS.items():
        for order in (1, 2):
            for budget in BUDGETS:
                group = [r for r in rows if (r["metric"], r["order"], r["budget"]) == (metric, order, budget)]
                assert sorted(r["seed"] for r in group) == [0, 1, 2]
                stats = {"metric": metric, "order": order, "budget": budget}
                for key in ("inrun", "graph", "delta"):
                    values = [r[key] for r in group]
                    stats[key] = {"runs": values, "mean": float(np.mean(values)), "std": float(np.std(values))}
                stats["graph_win"] = direction * stats["delta"]["mean"] > 1e-12
                groups.append(stats)
    counts = {metric: sum(r["graph_win"] for r in groups if r["metric"] == metric) for metric in METRICS}
    chosen = [metric for metric in METRICS if counts[metric] > 4]
    protocol = {
        "budgets_percent": list(BUDGETS), "seeds": [0, 1, 2], "std_ddof": 0,
        "comparison": "Graph Adam-aware + diversity vs corresponding-order In-Run Top-k",
        "focus_rule": "Post hoc: strictly better three-seed mean in more than 4 of 8 order-budget settings",
        "metric_directions": {k: v[1] for k, v in METRICS.items()},
        "interpretation": "Win counts are descriptive, not statistical significance. Domain entropy/count measure spread, not downstream utility.",
        "implementation_boundary": "Both use project-local approximations; see pile_main_100k/results/baseline_reproduction_audit.md.",
    }
    (out / "comparison.json").write_text(json.dumps({"protocol": protocol, "wins_out_of_8": counts,
                                                  "focused_metrics": chosen, "groups": groups}, indent=2, allow_nan=False) + "\n")
    with (out / "per_seed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for focused in (False, True):
        metrics = chosen if focused else list(METRICS)
        lines = ["# Graph vs In-Run Data Shapley", "",
                 "Graph is the full Graph Adam-aware + diversity arm; In-Run uses Top-k with the matching order.",
                 "Budgets: 1/2/5/10%; seeds: 0/1/2. Mean (population std).", ""]
        if focused:
            lines.extend(["Post-hoc focused display: metrics are selected because Graph wins on the three-seed mean in more than 4/8 settings. All four budgets and both orders are retained. See full_comparison.md for all six evaluated metrics. This is not a claim of overall or statistically significant superiority.", ""])
        for metric in metrics:
            label, direction = METRICS[metric]
            lines.extend([f"## {label} ({'higher' if direction > 0 else 'lower'} is the comparison direction)", "",
                          f"Graph wins: {counts[metric]}/8 settings.", "",
                          "| Order | Method | 1% | 2% | 5% | 10% |", "|---|---|---:|---:|---:|---:|"])
            for order in (1, 2):
                for method in ("inrun", "graph"):
                    values = []
                    for budget in BUDGETS:
                        item = next(r for r in groups if (r["metric"], r["order"], r["budget"]) == (metric, order, budget))
                        stats = item[method]
                        value = f'{stats["mean"]:.4f} ({stats["std"]:.3f})'
                        if (method == "graph" and item["graph_win"]) or (method == "inrun" and direction * item["delta"]["mean"] < -1e-12):
                            value = f"**{value}**"
                        values.append(value)
                    lines.append(f"| {order} | {method} | " + " | ".join(values) + " |")
            lines.append("")
        (out / ("focused_comparison.md" if focused else "full_comparison.md")).write_text("\n".join(lines) + "\n")
    (out / "source_manifest.json").write_text(json.dumps({str(p.relative_to(ROOT)): digest(p) for p in sorted(sources)}, indent=2) + "\n")
    print(json.dumps({"wins_out_of_8": counts, "focused_metrics": chosen}, indent=2))


if __name__ == "__main__":
    main()
