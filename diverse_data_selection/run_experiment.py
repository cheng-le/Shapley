"""Four-arm data-selection diagnostics from frozen, matched Pile caches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from pile_main_100k.common import diversity_select, symmetric_graph

ROOT = Path(__file__).resolve().parents[1]
PILE = ROOT / "pile_main_100k"
ARMS = ("Adam Top-k", "Graph Top-k", "Adam + diversity", "Graph + diversity")
METRICS = ("affinity_per_item", "graph_coverage", "domain_entropy",
           "domain_count", "input_tokens", "training_targets", "topk_overlap")


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def index_digest(selected):
    # Match the original runner's dtype-sensitive selection fingerprint.
    return hashlib.sha256(np.sort(selected).tobytes()).hexdigest()


def utility_cell(seed, budget, order, arm, selected, output):
    tag = f"seed{seed}_order{order}_budget{budget}_{arm.replace(' ', '_').replace('+', 'plus')}"
    local = output / "raw" / f"{tag}.json"
    if local.exists():
        cell = json.loads(local.read_text())
        expected = {"seed": seed, "budget": budget, "order": order, "arm": arm,
                    "indices_sha256": index_digest(selected), "epochs": 1,
                    "eval_part": "evaluation", "lr_schedule": "constant", "lr": 6e-5,
                    "batch_size": 16}
        if any(cell.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Local utility protocol mismatch: {local}")
        return {"val_loss": cell["val_loss"], "accuracy": cell["accuracy"],
                "document_weighted_ppl_proxy": float(np.exp(cell["val_loss"])),
                "source": str(local.resolve().relative_to(ROOT))}
    if arm not in ("Adam Top-k", "Graph + diversity") or budget not in (1, 2, 5, 10):
        return None
    method_index = (8 if arm == "Adam Top-k" else 10) + order - 1
    path = PILE / "low_ratio_validation_loss/raw" / f"method{method_index}_seed{seed}_budget{budget}.json"
    if not path.exists():
        return None
    cell = json.loads(path.read_text())
    prefix = "Graph " if arm.startswith("Graph") else ""
    method = f"{prefix}Adam-aware {order}{'st' if order == 1 else 'nd'} Order In-Run"
    strategy = "top_k" if arm == "Adam Top-k" else "symmetric_graph_greedy"
    expected = {"seed": seed, "budget": budget, "method": method,
                "selected_corpora": len(selected), "epochs": 1,
                "eval_part": "evaluation", "lr_schedule": "constant",
                "protocol_split": "pile_main_100k/protocol_v2_split.json",
                "selected_indices_sha256": index_digest(selected)}
    if any(cell.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Utility protocol/selection mismatch: {path}")
    if cell["selection"]["strategy"] != strategy:
        raise ValueError(f"Selection strategy mismatch: {path}")
    return {"val_loss": cell["val_loss"], "accuracy": cell["accuracy"],
            "document_weighted_ppl_proxy": float(np.exp(cell["val_loss"])),
            "source": str(path.relative_to(ROOT))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 5, 10])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "diverse_data_selection/results")
    args = parser.parse_args()
    if len(set(args.budgets)) != len(args.budgets) or any(b <= 0 or b >= 100 for b in args.budgets):
        parser.error("Budgets must be distinct percentages between 0 and 100")
    out = args.output_dir
    (out / "selected").mkdir(parents=True, exist_ok=True)
    metadata_path = PILE / "token_cache/metadata.json"
    fingerprint = json.loads(metadata_path.read_text())["fingerprint"]
    masks = np.load(PILE / "token_cache/train_mask.npy", mmap_mode="r")
    sources = {metadata_path, PILE / "protocol_v2_split.json", Path(__file__),
               PILE / "common.py", PILE / "selection_accuracy_cached_runner.py",
               Path(__file__).with_name("train_cell.py"),
               Path(__file__).with_name("run_training.py")}
    rows, pending = [], []
    for seed in range(3):
        raw_path = PILE / f"cache/clean_seed{seed}.npz"
        graph_path = PILE / f"cache_graph_v2/clean_seed{seed}_graph_v2.npz"
        sources.update((raw_path, graph_path))
        with np.load(raw_path, allow_pickle=False) as raw, np.load(graph_path, allow_pickle=False) as graph:
            if any(str(cache["fingerprint"]) != fingerprint for cache in (raw, graph)):
                raise ValueError("Token and score fingerprints differ")
            neighbors, weights = graph["graph_neighbors"], graph["graph_weights"]
            if not np.array_equal(neighbors, raw["graph_neighbors"]) or not np.array_equal(weights, raw["graph_weights"]):
                raise ValueError("Arms must share the same semantic graph")
            n = len(neighbors)
            if n != len(masks):
                raise ValueError("Score and token row counts differ")
            adjacency = symmetric_graph(neighbors, weights)
            domains = raw["domains"]
            for order in (1, 2):
                name = f"Adam-aware {order}{'st' if order == 1 else 'nd'} Order In-Run"
                scores_by_type = {"Adam": raw[f"score::{name}"], "Graph": graph[f"score::Graph {name}"]}
                for scores in scores_by_type.values():
                    if scores.shape != (n,) or not np.isfinite(scores).all():
                        raise ValueError("Invalid score vector")
                for budget in args.budgets:
                    count = round(n * budget / 100)
                    for arm in ARMS:
                        scores = scores_by_type[arm.split()[0]]
                        top = np.argpartition(scores, -count)[-count:]
                        if "+ diversity" in arm:
                            selected, _ = diversity_select(scores, neighbors, weights, count, 0.05)
                        else:
                            selected = top
                        if len(np.unique(selected)) != count:
                            raise ValueError("Duplicate or missing selected indices")
                        mask = np.zeros(n, dtype=bool)
                        mask[selected] = True
                        covered = mask.copy()
                        covered[adjacency[mask].indices] = True
                        _, domain_counts = np.unique(domains[selected], return_counts=True)
                        probabilities = domain_counts / count
                        tag = f"seed{seed}_order{order}_budget{budget}_{arm.replace(' ', '_').replace('+', 'plus')}"
                        selected_path = out / "selected" / f"{tag}.npy"
                        np.save(selected_path, selected)
                        row = {"seed": seed, "order": order, "budget": budget, "arm": arm,
                               "selected_count": count, "indices_sha256": index_digest(selected),
                               "affinity_per_item": float(adjacency[mask][:, mask].sum()) / (2 * count),
                               "graph_coverage": float(covered.mean()),
                               "domain_entropy": float(-np.sum(probabilities * np.log(probabilities))),
                               "domain_count": len(domain_counts),
                               "input_tokens": int(masks[selected].sum()),
                               "training_targets": int(masks[selected, 1:].sum()),
                               "topk_overlap": float(np.intersect1d(selected, top).size / count)}
                        cell = utility_cell(seed, budget, order, arm, selected, out)
                        row.update({key: cell[key] if cell else None for key in
                                    ("val_loss", "accuracy", "document_weighted_ppl_proxy", "source")})
                        if cell:
                            sources.add(ROOT / cell["source"])
                        else:
                            pending.append({"seed": seed, "order": order, "budget": budget,
                                            "arm": arm, "selected_indices": str(selected_path),
                                            "indices_sha256": row["indices_sha256"]})
                        rows.append(row)
                    print(f"Completed seed={seed} order={order} budget={budget}", flush=True)
    summary = []
    for order in (1, 2):
        for budget in args.budgets:
            for arm in ARMS:
                group = [r for r in rows if (r["order"], r["budget"], r["arm"]) == (order, budget, arm)]
                item = {"order": order, "budget": budget, "arm": arm}
                for metric in (*METRICS, "val_loss", "accuracy", "document_weighted_ppl_proxy"):
                    values = [r[metric] for r in group if r[metric] is not None]
                    item[metric] = {"runs": values, "mean": float(np.mean(values)), "std": float(np.std(values))} if len(values) == 3 else None
                summary.append(item)
    with (out / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    contrasts = (("Graph Top-k", "Adam Top-k"),
                 ("Adam + diversity", "Adam Top-k"),
                 ("Graph + diversity", "Graph Top-k"),
                 ("Graph + diversity", "Adam Top-k"))
    effects = []
    for order in (1, 2):
        for budget in args.budgets:
            for treatment, reference in contrasts:
                effect = {"order": order, "budget": budget,
                          "treatment": treatment, "reference": reference}
                for metric in ("val_loss", "affinity_per_item", "graph_coverage"):
                    differences = []
                    for seed in range(3):
                        pair = {r["arm"]: r for r in rows if
                                (r["order"], r["budget"], r["seed"]) == (order, budget, seed)}
                        left, right = pair[treatment][metric], pair[reference][metric]
                        if left is not None and right is not None:
                            differences.append(left - right)
                    effect[metric] = {"paired_deltas": differences,
                                      "mean_delta": float(np.mean(differences)),
                                      "std_delta": float(np.std(differences))} if len(differences) == 3 else None
                effects.append(effect)
    (out / "paired_effects.json").write_text(json.dumps(effects, indent=2, allow_nan=False) + "\n")
    (out / "pending_evaluations.json").write_text(json.dumps(pending, indent=2) + "\n")
    manifest = {str(p.relative_to(ROOT)): digest(p) for p in sorted(sources)}
    (out / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    lines = ["# Four-arm Selection Diagnostics", "", "Population std over seeds 0/1/2. Missing utility evaluations are marked pending.", "",
             "| Order | Budget | Arm | Affinity/item | Graph coverage | Domain entropy | Validation loss |",
             "|---|---:|---|---:|---:|---:|---:|"]
    for item in summary:
        cells = []
        for metric in ("affinity_per_item", "graph_coverage", "domain_entropy", "val_loss"):
            stats = item[metric]
            cells.append(f'{stats["mean"]:.4f} ({stats["std"]:.3f})' if stats else "pending")
        lines.append(f'| {item["order"]} | {item["budget"]}% | {item["arm"]} | ' + " | ".join(cells) + " |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Saved {len(rows)} selection cells; {len(pending)} utility evaluations pending: {out}")


if __name__ == "__main__":
    main()
