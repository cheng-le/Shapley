"""Appendix E diagnostics on fixed, previously observed CIFAR trajectories."""
from __future__ import annotations
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
spec = importlib.util.spec_from_file_location("appendix_d", ROOT / "appendix_d_cifar/run.py")
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)
import numpy as np
import torch
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from torchvision.datasets import CIFAR10
from common import confidence_gated_graph_weights, diversity_select

OUT = HERE / "results"
SOURCE = ROOT / "appendix_d_cifar"
METHODS = ["k10_a0_l0", "full", "no_gate", "no_history", "no_pair", "no_scale", "random"]
LABELS = ["Adam first", "Graph gated", "No gate", "No history", "No pair", "No scale", "Random"]


def csv_write(name, rows):
    d.csv_write(OUT / name, rows)


def percentile(scores):
    return 1 - (rankdata(-scores, method="average") - 1) / (len(scores) - 1)


def auc(labels, scores):
    return float(roc_auc_score(labels, -scores)) if len(np.unique(labels)) == 2 else None


def propagation(trace, nn, weights, confidence, noise, block_noise=False):
    """Allocate centered messages at the factual denominator; also recompute a counterfactual."""
    n = len(nn)
    history = np.zeros(n, dtype=np.float32)
    scores = np.zeros(n, dtype=np.float32)
    allocations = np.zeros((4, n), dtype=np.float32)
    step_rows = []
    for step, (a, b) in enumerate(zip(trace["ptr"][:-1], trace["ptr"][1:])):
        ids, raw = trace["ids"][a:b], trace["raw"][a:b]
        neighbors, ww = nn[ids], weights[ids]
        local = np.zeros(n, dtype=np.float32)
        local[ids] = raw
        parts = []
        for values in (history, local):
            clean = (ww * values[neighbors] * (1 - noise[neighbors])).sum(1) * confidence[ids]
            corrupt = (ww * values[neighbors] * noise[neighbors]).sum(1) * confidence[ids]
            if block_noise:
                corrupt = np.zeros_like(corrupt)
            parts.extend((clean, corrupt))
        matched, ratios = [], []
        for offset, strength in ((0, .2), (2, .1)):
            total = parts[offset] + parts[offset + 1]
            raw_sd, term_sd = float(raw.std()), float(total.std())
            ratios.append(term_sd / max(raw_sd, 1e-12))
            factor = strength * raw_sd / (term_sd + 1e-12) if min(raw_sd, term_sd) >= 1e-12 else 0.
            for part in parts[offset:offset + 2]:
                matched.append((part - float(part.mean())) * factor)
        corrected = raw + sum(matched)
        scores[ids] += corrected
        for i, values in enumerate(matched):
            allocations[i, ids] += values
        history[ids] += raw
        step_rows.append(dict(step=step, history_to_raw_sd=ratios[0], pair_to_raw_sd=ratios[1],
            noise_history_l1=float(np.abs(parts[1]).sum()), clean_history_l1=float(np.abs(parts[0]).sum()),
            noise_pair_l1=float(np.abs(parts[3]).sum()), clean_pair_l1=float(np.abs(parts[2]).sum())))
    return scores, allocations, step_rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    dataset = CIFAR10(str(ROOT / "data/CIFAR10"), train=True, download=False)
    true_labels = np.asarray(dataset.targets)
    class_names = dataset.classes
    nodes, edges, contamination, distributions, buckets, effects, steps, correlations = [], [], [], [], [], [], [], []
    figures = {}
    case_pool = None
    hashes = {}
    max_replay_error = 0.
    for ds in range(3):
        path = SOURCE / "data" / ("seed%d.npz" % ds)
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as source:
            features, observed, noise, source_ids = [source[k] for k in ("resnet128", "labels", "noise", "source_indices")]
        clean_labels = true_labels[source_ids]
        np.testing.assert_array_equal((observed != clean_labels).astype(np.int8), noise)
        nn, ww, _, dense = d.graph(features, 10)
        symmetric = (dense + dense.T) * .5
        degree = (symmetric > 0).sum(1)
        unit = features / np.linalg.norm(features, axis=1, keepdims=True).clip(1e-8)
        density = (unit[:, None, :] * unit[nn]).sum(2).mean(1)
        observed_agreement = (ww * (observed[:, None] == observed[nn])).sum(1)
        true_agreement = (ww * (clean_labels[:, None] == clean_labels[nn])).sum(1)
        incoming_noise = (ww * noise[nn]).sum(1)
        all_vectors = []
        vector_names = []
        for ts in range(3):
            folder = SOURCE / "results" / ("data%d_train%d" % (ds, ts))
            for name in ("scores.npz", "trajectory.npz"):
                path = folder / name
                hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(folder / "scores.npz") as cache:
                vectors = {m: cache[m] for m in METHODS}
            with np.load(folder / "trajectory.npz") as cache:
                trace = {k: cache[k] for k in cache.files}
            gated, confidence = confidence_gated_graph_weights(trace["initial_grad"], nn, ww, chunk_size=32)
            factual, allocation, step_rows = propagation(trace, nn, gated, confidence, noise)
            np.testing.assert_allclose(factual, vectors["full"], atol=1e-8, rtol=5e-5)
            max_replay_error = max(max_replay_error, float(np.max(np.abs(factual - vectors["full"]))))
            np.testing.assert_allclose(factual - vectors["k10_a0_l0"], allocation.sum(0), atol=1e-8, rtol=5e-5)
            blocked, _, _ = propagation(trace, nn, gated, confidence, noise, block_noise=True)
            # Removing no sources is the factual path; removing all emission leaves raw scores.
            if ds == ts == 0:
                none_removed = propagation(trace, nn, gated, confidence, np.zeros(1000, dtype=np.int8), True)[0]
                all_removed = propagation(trace, nn, gated, confidence, np.ones(1000, dtype=np.int8), True)[0]
                np.testing.assert_allclose(none_removed, factual, atol=1e-8, rtol=5e-5)
                np.testing.assert_array_equal(all_removed, vectors["k10_a0_l0"])
            steps.extend(dict(data_seed=ds, train_seed=ts, **r) for r in step_rows)
            raw = vectors["k10_a0_l0"]
            raw_pct, graph_pct = percentile(raw), percentile(vectors["full"])
            blocked_pct = percentile(blocked)
            grad = trace["initial_grad"]
            norms = np.linalg.norm(grad, axis=1).clip(1e-8)
            cosine = np.empty(nn.shape, dtype=np.float32)
            for a in range(0, 1000, 32):
                cosine[a:a + 32] = (grad[a:a + 32, None] * grad[nn[a:a + 32]]).sum(2) / (norms[a:a + 32, None] * norms[nn[a:a + 32]])
            gradient_agreement = (ww * cosine).sum(1)
            for i in range(1000):
                nodes.append(dict(data_seed=ds, train_seed=ts, local_id=i, source_id=int(source_ids[i]),
                    observed_label=int(observed[i]), true_label=int(clean_labels[i]), corrupted=int(noise[i]),
                    raw_score=float(raw[i]), graph_score=float(vectors["full"][i]),
                    raw_percentile=float(raw_pct[i]), graph_percentile=float(graph_pct[i]),
                    percentile_lift=float(graph_pct[i] - raw_pct[i]), blocked_percentile=float(blocked_pct[i]),
                    blocked_delta=float(blocked[i] - factual[i]), degree=int(degree[i]), raw_cosine_density=float(density[i]),
                    observed_label_agreement=float(observed_agreement[i]), true_label_agreement=float(true_agreement[i]),
                    signed_gradient_agreement=float(gradient_agreement[i]), gate_confidence=float(confidence[i]),
                    incoming_noise_weight=float(incoming_noise[i]),
                    clean_history_correction=float(allocation[0, i]), noise_history_correction=float(allocation[1, i]),
                    clean_pair_correction=float(allocation[2, i]), noise_pair_correction=float(allocation[3, i])))
                for j, neighbor in enumerate(nn[i]):
                    edges.append(dict(data_seed=ds, train_seed=ts, recipient=i, source=int(neighbor),
                        recipient_corrupted=int(noise[i]), source_corrupted=int(noise[neighbor]),
                        semantic_weight=float(ww[i, j]), gradient_cosine=float(cosine[i, j]),
                        gated_effective_weight=float(gated[i, j] * confidence[i])))
            for method, values in vectors.items():
                all_vectors.append(values)
                vector_names.append(method + "/s%d" % ts)
                z = (values - values.mean()) / max(float(values.std()), 1e-12)
                for i in range(1000):
                    distributions.append(dict(data_seed=ds, train_seed=ts, method=method, local_id=i,
                        corrupted=int(noise[i]), score=float(values[i]), within_seed_z=float(z[i])))
            augmented = dict(vectors, graph_block_noise=blocked)
            for method, values in augmented.items():
                for budget in (.01, .02, .05, .1, .2, .4):
                    selected = np.argsort(-values, kind="stable")[:round(1000 * budget)]
                    contamination.append(dict(data_seed=ds, train_seed=ts, method=method, budget=budget,
                        count=len(selected), corrupted_selected=int(noise[selected].sum()),
                        contamination=float(noise[selected].mean()), noise_auroc=auc(noise, values)))
            for budget in (.01, .02, .05, .1, .2, .4):
                selected, _ = diversity_select(vectors["full"], nn, ww, round(1000 * budget), lambda_fraction=.05)
                contamination.append(dict(data_seed=ds, train_seed=ts, method="graph_diverse", budget=budget,
                    count=len(selected), corrupted_selected=int(noise[selected].sum()),
                    contamination=float(noise[selected].mean()), noise_auroc=auc(noise, vectors["full"])))
            for status in (0, 1):
                mask = noise == status
                effects.append(dict(data_seed=ds, train_seed=ts, corrupted=status, count=int(mask.sum()),
                    promoted_fraction=float((graph_pct[mask] > raw_pct[mask]).mean()),
                    mean_percentile_lift=float((graph_pct[mask] - raw_pct[mask]).mean()),
                    mean_blocked_rank_change=float((blocked_pct[mask] - graph_pct[mask]).mean()),
                    mean_absolute_blocked_score_change=float(np.abs(blocked[mask] - factual[mask]).mean()),
                    noise_source_abs_allocation=float(np.abs(allocation[[1, 3]][:, mask]).sum()),
                    clean_source_abs_allocation=float(np.abs(allocation[[0, 2]][:, mask]).sum())))
            top_mask = np.zeros(1000, dtype=bool)
            top_mask[np.argsort(-vectors["full"], kind="stable")[:200]] = True
            for variable, values in (("label_agreement", observed_agreement), ("gradient_agreement", gradient_agreement),
                                     ("degree", degree), ("density", density)):
                cuts = np.unique(np.quantile(values, np.linspace(0, 1, 6)))
                assignment = np.digitize(values, cuts[1:-1], right=True)
                for bucket in np.unique(assignment):
                    mask = assignment == bucket
                    selected = mask & top_mask
                    buckets.append(dict(data_seed=ds, train_seed=ts, variable=variable, bucket=int(bucket),
                        lower=float(values[mask].min()), upper=float(values[mask].max()), count=int(mask.sum()),
                        corrupted_count=int(noise[mask].sum()), noise_fraction=float(noise[mask].mean()),
                        selected_count=int(selected.sum()), selected_noise_fraction=float(noise[selected].mean()) if selected.any() else None,
                        mean_percentile_lift=float((graph_pct[mask] - raw_pct[mask]).mean()),
                        noise_auroc=auc(noise[mask], vectors["full"][mask])))
            figures[ds, ts] = (raw.copy(), vectors["full"].copy(), noise.copy())
            if ds == ts == 0:
                case_pool = dict(source_ids=source_ids, noise=noise, observed=observed, truth=clean_labels,
                    nn=nn, ww=ww, gated=gated, confidence=confidence, raw=raw, graph=vectors["full"],
                    shift=graph_pct - raw_pct, graph_pct=graph_pct, raw_pct=raw_pct)
            np.savez_compressed(OUT / ("counterfactual_d%d_t%d.npz" % (ds, ts)), factual=factual,
                block_noise_sources=blocked, allocations=allocation)
        matrix = spearmanr(np.stack(all_vectors), axis=1).correlation
        np.savez_compressed(OUT / ("rank_matrix_data%d.npz" % ds), matrix=matrix, labels=np.asarray(vector_names))
        for i, a in enumerate(vector_names):
            for j, b in enumerate(vector_names):
                correlations.append(dict(data_seed=ds, left=a, right=b, spearman=float(matrix[i, j])))
        assert np.allclose(matrix, matrix.T) and np.allclose(np.diag(matrix), 1)
        print("Finished dataset", ds, flush=True)
    for name, rows in (("nodes", nodes), ("edges", edges), ("contamination", contamination),
                       ("score_distribution", distributions), ("buckets", buckets), ("propagation_effects", effects),
                       ("step_messages", steps), ("rank_correlations", correlations)):
        csv_write(name + ".csv", rows)
    contamination_summary = []
    for method in METHODS + ["graph_block_noise", "graph_diverse"]:
        for budget in (.01, .02, .05, .1, .2, .4):
            group = [r for r in contamination if r["method"] == method and r["budget"] == budget]
            contamination_summary.append(dict(method=method, budget=budget,
                mean=float(np.mean([r["contamination"] for r in group])),
                std=float(np.std([r["contamination"] for r in group], ddof=1)),
                mean_auroc=float(np.mean([r["noise_auroc"] for r in group]))))
    csv_write("contamination_summary.csv", contamination_summary)
    stable = []
    for method in METHODS:
        for ds in range(3):
            pairs = [next(r["spearman"] for r in correlations if r["data_seed"] == ds and
                r["left"] == method + "/s%d" % a and r["right"] == method + "/s%d" % b)
                for a, b in ((0, 1), (0, 2), (1, 2))]
            stable.append(dict(method=method, data_seed=ds, mean_pairwise_spearman=float(np.mean(pairs))))
    csv_write("stability_summary.csv", stable)
    cases = []
    for status, operation, title in ((1, np.argmin, "Corrupt: largest demotion"), (1, np.argmax, "Corrupt: largest promotion"),
                                     (0, np.argmax, "Clean: largest promotion"), (0, np.argmin, "Clean: largest demotion")):
        candidates = np.flatnonzero(case_pool["noise"] == status)
        i = int(candidates[operation(case_pool["shift"][candidates])])
        positions = np.argsort(-case_pool["ww"][i], kind="stable")[:5]
        neighbors = case_pool["nn"][i, positions]
        cases.append(dict(title=title, data_seed=0, train_seed=0, local_id=i,
            source_id=int(case_pool["source_ids"][i]), corrupted=status,
            observed_label=class_names[int(case_pool["observed"][i])], true_label=class_names[int(case_pool["truth"][i])],
            raw_rank=int(rankdata(-case_pool["raw"], method="min")[i]),
            graph_rank=int(rankdata(-case_pool["graph"], method="min")[i]),
            percentile_lift=float(case_pool["shift"][i]),
            neighbors=[dict(local_id=int(j), source_id=int(case_pool["source_ids"][j]),
                observed_label=class_names[int(case_pool["observed"][j])], true_label=class_names[int(case_pool["truth"][j])],
                corrupted=int(case_pool["noise"][j]), semantic_weight=float(case_pool["ww"][i, p]),
                gated_effective_weight=float(case_pool["gated"][i, p] * case_pool["confidence"][i])) for p, j in zip(positions, neighbors)]))
    d.dump(OUT / "cases.json", cases)
    plots(figures, contamination_summary, stable, buckets, cases, dataset)
    d.dump(OUT / "provenance.json", dict(source_hashes=hashes,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        graph_code_sha256=hashlib.sha256((SOURCE / "run.py").read_bytes()).hexdigest(),
        source="Appendix D frozen-ResNet, initial-gradient gated extension",
        methods=METHODS, no_retraining=True, noise_labels_diagnostic_only=True))
    d.dump(OUT / "validation.json", dict(status="passed", datasets=3, trajectories=9,
        node_rows=len(nodes), edge_rows=len(edges), distribution_rows=len(distributions),
        contamination_cells=len(contamination), rank_correlation_cells=len(correlations),
        max_replay_absolute_error=max_replay_error, counterfactual_endpoint_tests="passed",
        message_allocation_reconstruction="passed", corruption_matches_original_labels=True))


def plots(figures, contamination, stability, buckets, cases, dataset):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "svg.fonttype": "none"})
    def save(fig, name):
        fig.tight_layout()
        for suffix in ("png", "pdf", "svg"):
            fig.savefig(OUT / (name + "." + suffix), dpi=180)
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ds, ax in enumerate(axes):
        with np.load(OUT / ("rank_matrix_data%d.npz" % ds)) as z:
            m = z["matrix"]
        ax.imshow(m, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set(xticks=range(21), yticks=range(21), xticklabels=["%s/s%d" % (LABELS[i % 7], i // 7) for i in range(21)],
               yticklabels=["%s/s%d" % (LABELS[i % 7], i // 7) for i in range(21)], title="Data seed %d; Spearman" % ds)
        ax.tick_params(labelsize=6, axis="x", labelrotation=90)
        ax.tick_params(labelsize=6, axis="y")
    save(fig, "rank_matrices")
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    for (ds, ts), (raw, graph, noise) in figures.items():
        ax = axes[ds, ts]
        values = []
        for scores in (raw, graph):
            z = (scores - scores.mean()) / max(float(scores.std()), 1e-12)
            values.extend([z[noise == 0], z[noise == 1]])
        ax.violinplot(values, showmedians=True, showextrema=False)
        ax.set(xticks=[1, 2, 3, 4], xticklabels=["Raw clean", "Raw noise", "Graph clean", "Graph noise"],
               ylabel="Within-seed z score", title="Data %d / train %d" % (ds, ts))
        ax.tick_params(axis="x", labelrotation=25, labelsize=7)
    save(fig, "per_seed_distributions")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for method in ("k10_a0_l0", "full", "no_gate", "no_scale", "random", "graph_block_noise"):
        group = [r for r in contamination if r["method"] == method]
        axes[0].errorbar([r["budget"] * 100 for r in group], [r["mean"] for r in group],
            yerr=[r["std"] for r in group], marker="o", capsize=2, label=method, linewidth=1)
    axes[0].axhline(.1, color="gray", linestyle="--", linewidth=.8)
    axes[0].set(xlabel="Top-k retention (%)", ylabel="Selected noise fraction")
    axes[0].legend(fontsize=6)
    values = [np.mean([r["mean_pairwise_spearman"] for r in stability if r["method"] == m]) for m in METHODS]
    axes[1].bar(range(7), values, color="#287e69")
    axes[1].set(xticks=range(7), xticklabels=LABELS, ylabel="Mean cross-training-seed Spearman")
    axes[1].tick_params(axis="x", labelrotation=35)
    save(fig, "contamination_stability")
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5))
    for ax, variable in zip(axes, ("label_agreement", "gradient_agreement", "degree", "density")):
        group = [r for r in buckets if r["variable"] == variable]
        for ds in range(3):
            r = [x for x in group if x["data_seed"] == ds and x["train_seed"] == 0]
            ax.plot([x["bucket"] + 1 for x in r], [x["noise_fraction"] for x in r], "o-", label="data%d/train0" % ds)
        ax.set(xlabel="Quantile bin (ties merged)", ylabel="Corruption fraction", title=variable)
    axes[0].legend(fontsize=6)
    save(fig, "neighborhood_buckets")
    fig, axes = plt.subplots(4, 6, figsize=(13, 10))
    for row, case in enumerate(cases):
        items = [case] + case["neighbors"]
        for col, item in enumerate(items):
            ax = axes[row, col]
            ax.imshow(dataset.data[item["source_id"]])
            ax.set_xticks([])
            ax.set_yticks([])
            title = "id=%d; %s\ntrue=%s / obs=%s" % (item["source_id"], "noise" if item["corrupted"] else "clean", item["true_label"], item["observed_label"])
            if col == 0:
                title = case["title"] + "\n" + title
                ax.set_xlabel("Rank %d -> %d" % (case["raw_rank"], case["graph_rank"]))
            else:
                ax.set_xlabel("w=%.3f; gated=%.3f" % (item["semantic_weight"], item["gated_effective_weight"]))
            ax.set_title(title, fontsize=7)
    save(fig, "neighborhood_cases")


if __name__ == "__main__":
    main()
