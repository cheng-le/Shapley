"""Cached-score efficiency and mechanism diagnostics; not end-to-end training."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE / "vendor"))
sys.path.insert(0, str(ROOT / "pile_main_100k"))
import faiss
import numpy as np
import psutil
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from common import (confidence_gated_graph_weights, diversity_select,
                    replay_graph_trajectory, symmetric_graph)
from graph_utils import gpu_chunked_knn

OUT = HERE / "results"
CACHE = ROOT / "pile_main_100k" / "cache"
DEVICE = torch.device("cuda:2")
MIB = 1024 ** 2


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def subset(n):
    return np.random.default_rng(2027).permutation(100000)[:n]


def measured(fn):
    gc.collect()
    torch.cuda.synchronize(DEVICE)
    torch.cuda.reset_peak_memory_stats(DEVICE)
    gpu0 = torch.cuda.memory_allocated(DEVICE)
    proc = psutil.Process()
    rss0 = proc.memory_info().rss
    peak = [rss0]
    stop = threading.Event()

    def sample():
        while not stop.wait(.005):
            peak[0] = max(peak[0], proc.memory_info().rss)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    start = time.perf_counter()
    try:
        result = fn()
        torch.cuda.synchronize(DEVICE)
        elapsed = time.perf_counter() - start
    finally:
        stop.set()
        thread.join()
    peak[0] = max(peak[0], proc.memory_info().rss)
    return result, dict(seconds=elapsed, rss_peak_mib=peak[0] / MIB,
                        rss_delta_mib=(peak[0] - rss0) / MIB,
                        cuda_extra_peak_mib=(torch.cuda.max_memory_allocated(DEVICE) - gpu0) / MIB)


def ann_knn(features, k=10):
    vectors = np.ascontiguousarray(features, dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True).clip(1e-8)
    index = faiss.IndexHNSWFlat(vectors.shape[1], 32)
    index.hnsw.efConstruction = 80
    index.hnsw.efSearch = 64
    index.add(vectors)
    distances, candidates = index.search(vectors, k + 1)
    neighbors = np.empty((len(vectors), k), dtype=np.int32)
    similarities = np.empty((len(vectors), k), dtype=np.float32)
    for i, row in enumerate(candidates):
        valid = (row != i) & (row >= 0)
        if valid.sum() < k:
            raise RuntimeError("ANN returned too few non-self neighbors")
        neighbors[i] = row[valid][:k]
        similarities[i] = np.maximum(1 - distances[i][valid][:k] / 2, 0)
    similarities /= similarities.sum(1, keepdims=True).clip(1e-8)
    return neighbors, similarities, index


def replay(raw, neighbors, weights, confidence, seed, alpha=.2,
           include_pair=False, historical=True, scale=True):
    """Local ablation wrapper; defaults reproduce the existing replay exactly."""
    raw = np.asarray(raw, dtype=np.float32)
    history = np.zeros_like(raw)
    corrected = np.zeros_like(raw)
    order = np.random.default_rng(seed).permutation(len(raw))

    def transform(term, values):
        if not scale:
            return term
        a, b = float(values.std()), float(term.std())
        if a < 1e-12 or b < 1e-12:
            return np.zeros_like(term)
        return (term - float(term.mean())) * (a / (b + 1e-12))

    for start in range(0, len(order), 8):
        ids = order[start:start + 8]
        values = raw[ids]
        nn, ww = neighbors[ids], weights[ids]
        diffusion = (ww * history[nn]).sum(1) * confidence[ids]
        corrected_values = values + alpha * transform(diffusion, values)
        if include_pair:
            positions = {int(index): pos for pos, index in enumerate(ids)}
            pair = np.zeros(len(ids), dtype=np.float32)
            for row, (row_nn, row_ww) in enumerate(zip(nn, ww)):
                for neighbor, weight in zip(row_nn, row_ww):
                    position = positions.get(int(neighbor))
                    if position is not None:
                        pair[row] += float(weight) * float(values[position])
            pair *= confidence[ids]
            corrected_values += .5 * alpha * transform(pair, values)
        corrected[ids] = corrected_values
        if historical:
            history[ids] += values
    return corrected


def pointwise_replay(raw, seed=0):
    result = np.zeros_like(raw)
    order = np.random.default_rng(seed).permutation(len(raw))
    for start in range(0, len(raw), 8):
        ids = order[start:start + 8]
        result[ids] += raw[ids]
    return result


def benchmark(n, repetition):
    ids = subset(n)
    with np.load(CACHE / "clean_seed0.npz") as cache:
        embeddings = cache["embeddings"][ids]
        gradients = cache["gradients"][ids]
        raw = [cache["score::Adam-aware %s Order In-Run" % order][ids]
               for order in ("1st", "2nd")]
    gpu_chunked_knn(embeddings[:256], DEVICE, k=10)
    rows = []
    def record(stage, fn, **extra):
        result, timing = measured(fn)
        rows.append(dict(n=n, repetition=repetition, stage=stage, **timing, **extra))
        print(n, repetition, stage, round(timing["seconds"], 3), flush=True)
        return result
    record("pointwise_replay", lambda: pointwise_replay(raw[0]))
    record("pointwise_top10", lambda: np.argsort(-raw[0], kind="stable")[:n // 10])
    graphs = {}
    # Alternate construction order to reduce systematic warm-cache bias.
    for mode in (("exact", "ann") if repetition % 2 == 0 else ("ann", "exact")):
        if mode == "exact":
            graphs[mode] = record("exact_knn", lambda: gpu_chunked_knn(embeddings, DEVICE, k=10))
        else:
            nn, ww, index = record("ann_knn", lambda: ann_knn(embeddings))
            index_bytes = int(faiss.serialize_index(index).nbytes)
            del index
            graphs[mode] = nn, ww
    exact = graphs["exact"][0]
    approx = graphs["ann"][0]
    recall = float(np.mean([len(set(a).intersection(b)) / 10 for a, b in zip(exact, approx)]))
    scores = {}
    selections = {}
    for mode, (nn, ww) in graphs.items():
        gated, confidence = record(mode + "_gate", lambda: confidence_gated_graph_weights(gradients, nn, ww))
        for order in (1, 2):
            values = record(mode + "_replay%d" % order,
                            lambda: replay_graph_trajectory(raw[order - 1], nn, gated, confidence, 0,
                                                            include_pair=order == 2),
                            replay_batches=math.ceil(n / 8))
            scores[mode, order] = values
            selected, _ = record(mode + "_select%d" % order,
                                  lambda: diversity_select(values, nn, ww, n // 10, lambda_fraction=.05))
            selections[mode, order] = set(map(int, selected))
    dump(OUT / "timing" / ("n%d_rep%d.json" % (n, repetition)), dict(
        rows=rows, ann_recall10=recall, ann_index_bytes=index_bytes,
        sparse_graph_bytes=int(exact.nbytes + graphs["exact"][1].nbytes),
        graph_score_cache_bytes=int(exact.nbytes + graphs["exact"][1].nbytes * 2 + n * 4 * 3),
        input_tensor_bytes=int(embeddings.nbytes + gradients.nbytes + sum(v.nbytes for v in raw)),
        ann_score_spearman={str(o): float(spearmanr(scores["exact", o], scores["ann", o]).correlation)
                            for o in (1, 2)},
        ann_selection_overlap={str(o): len(selections["exact", o] & selections["ann", o]) / (n // 10)
                               for o in (1, 2)}))


def configurations():
    base = dict(k=10, alpha=.2, lam=.05, gate=True, historical=True,
                pair=True, scale=True, embedding="semantic")
    variants = [("full", {})]
    variants += [("k%d" % k, {"k": k}) for k in (5, 20)]
    variants += [("alpha%s" % a, {"alpha": a}) for a in (0, .1, .4)]
    variants += [("lambda%s" % v, {"lam": v}) for v in (0, .025, .1)]
    variants += [("no_" + flag, {flag: False}) for flag in ("gate", "historical", "pair", "scale")]
    variants += [("gradient_embedding", {"embedding": "gradient"})]
    return [(name, dict(base, **changes)) for name, changes in variants]


def ablation(seed):
    ids = subset(10000)
    records, clean_scores = [], {}
    for scenario in ("clean", "noise"):
        with np.load(CACHE / ("%s_seed%d.npz" % (scenario, seed))) as cache:
            embeddings = cache["embeddings"][ids]
            gradients = cache["gradients"][ids]
            labels = cache["noise_labels"][ids]
            domains = cache["domains"][ids]
            raw = [cache["score::Adam-aware %s Order In-Run" % order][ids]
                   for order in ("1st", "2nd")]
        semantic_nn, _ = gpu_chunked_knn(embeddings, DEVICE, k=20)
        gradient_nn, _ = gpu_chunked_knn(gradients, DEVICE, k=20)
        graphs = {}
        for embedding, features, nn in (("semantic", embeddings, semantic_nn),
                                        ("gradient", gradients, gradient_nn)):
            unit = features.astype(np.float32)
            unit /= np.linalg.norm(unit, axis=1, keepdims=True).clip(1e-8)
            # Recover similarities instead of renormalizing truncated weights.
            sims = np.empty(nn.shape, dtype=np.float32)
            for start in range(0, len(ids), 128):
                sims[start:start + 128] = np.maximum(np.einsum(
                    "nd,nkd->nk", unit[start:start + 128], unit[nn[start:start + 128]]), 0)
            for k in (5, 10, 20):
                w = sims[:, :k].copy()
                w /= w.sum(1, keepdims=True).clip(1e-8)
                g, c = confidence_gated_graph_weights(gradients, nn[:, :k], w)
                graphs[embedding, k] = nn[:, :k], w, g, c
        ref_nn, ref_w, _, _ = graphs["semantic", 10]
        reference = symmetric_graph(ref_nn, ref_w)
        for name, cfg in configurations():
            nn, ww, gated, confidence = graphs[cfg["embedding"], cfg["k"]]
            for order in (1, 2):
                values = replay(raw[order - 1], nn, gated if cfg["gate"] else ww,
                                confidence if cfg["gate"] else np.ones(len(ids), dtype=np.float32),
                                seed, alpha=cfg["alpha"], include_pair=order == 2 and cfg["pair"],
                                historical=cfg["historical"], scale=cfg["scale"])
                chosen, diagnostics = diversity_select(values, nn, ww, 1000, lambda_fraction=cfg["lam"])
                mask = np.zeros(len(ids), dtype=np.float64)
                mask[chosen] = 1
                counts = np.unique(domains[chosen], return_counts=True)[1]
                probabilities = counts / counts.sum()
                record = dict(seed=seed, scenario=scenario, variant=name, order=order, **cfg,
                              selected_noise_fraction=float(labels[chosen].mean()),
                              reference_pair_affinity=float(mask @ (reference @ mask) / 2 / 1000),
                              reference_coverage=float(((mask > 0) | (reference @ mask > 0)).mean()),
                              domain_count=len(counts), domain_entropy=float(-(probabilities * np.log(probabilities)).sum()),
                              score_spearman_vs_raw=float(spearmanr(values, raw[order - 1]).correlation))
                if scenario == "noise":
                    record["noise_auroc"] = float(roc_auc_score(labels, -values))
                else:
                    clean_scores[name + "__" + str(order)] = values
                records.append(record)
        print("ablations", seed, scenario, len(records), flush=True)
    dump(OUT / "ablation" / ("seed%d.json" % seed), records)
    np.savez_compressed(OUT / "ablation" / ("scores_seed%d.npz" % seed), **clean_scores)


def tests():
    rng = np.random.default_rng(19)
    features = rng.normal(size=(80, 12)).astype(np.float32)
    nn, ww = gpu_chunked_knn(features, DEVICE, k=5)
    unit = features / np.linalg.norm(features, axis=1, keepdims=True)
    sim = unit @ unit.T
    np.fill_diagonal(sim, -np.inf)
    expected = np.argsort(-sim, axis=1)[:, :5]
    np.testing.assert_array_equal(nn, expected)
    assert not np.any(nn == np.arange(80)[:, None])
    an, aw, _ = ann_knn(features, 5)
    assert not np.any(an == np.arange(80)[:, None])
    np.testing.assert_allclose(aw.sum(1), 1, atol=1e-6)
    raw = rng.normal(size=80).astype(np.float32)
    gated, confidence = confidence_gated_graph_weights(features, nn, ww)
    for pair in (False, True):
        for seed in (0, 1, 2):
            actual = replay(raw, nn, gated, confidence, seed, include_pair=pair)
            expected = replay_graph_trajectory(raw, nn, gated, confidence, seed, include_pair=pair)
            np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(replay(raw, nn, gated, confidence, 0, alpha=0), raw)
    np.testing.assert_array_equal(replay(raw, nn, gated, confidence, 0, historical=False), raw)
    selected, _ = diversity_select(raw, nn, ww, 8, lambda_fraction=0)
    assert set(selected) == set(np.argsort(-raw)[:8])
    assert np.array_equal(subset(1000), subset(10000)[:1000])
    dump(OUT / "tests.json", dict(status="passed", checks=["exact_vs_brute_force", "ANN_no_self",
         "ANN_weight_normalization", "replay_bitwise_parity_6_cases", "alpha_zero",
         "no_history_first_order", "lambda_zero_topk", "nested_subsets"]))
    print("Tests passed", flush=True)


def table(rows, path):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    timing_files = sorted((OUT / "timing").glob("*.json"))
    payloads = [json.loads(p.read_text()) for p in timing_files]
    timing = [r for p in payloads for r in p["rows"]]
    table(timing, OUT / "timing_all.csv")
    summary = []
    for n in (1000, 10000, 50000, 100000):
        for stage in sorted({r["stage"] for r in timing}):
            group = [r for r in timing if r["n"] == n and r["stage"] == stage]
            record = dict(n=n, stage=stage, repetitions=len(group))
            for key in ("seconds", "rss_peak_mib", "rss_delta_mib", "cuda_extra_peak_mib"):
                values = [r[key] for r in group]
                record[key + "_mean"] = float(np.mean(values))
                record[key + "_std"] = float(np.std(values, ddof=1))
            summary.append(record)
    table(summary, OUT / "timing_summary.csv")
    records = [r for p in sorted((OUT / "ablation").glob("seed*.json")) for r in json.loads(p.read_text())]
    table(records, OUT / "ablation_all.csv")
    scores = [np.load(OUT / "ablation" / ("scores_seed%d.npz" % s)) for s in range(3)]
    aggregate = []
    for name, cfg in configurations():
        for order in (1, 2):
            group = [r for r in records if r["variant"] == name and r["order"] == order]
            row = dict(variant=name, order=order)
            for scenario, metrics in (("noise", ["noise_auroc", "selected_noise_fraction"]),
                                      ("clean", ["reference_pair_affinity", "reference_coverage", "domain_entropy", "domain_count"])):
                for metric in metrics:
                    values = [r[metric] for r in group if r["scenario"] == scenario]
                    row[metric + "_mean"] = float(np.mean(values))
                    row[metric + "_std"] = float(np.std(values, ddof=1))
            key = name + "__" + str(order)
            correlations = [float(spearmanr(scores[a][key], scores[b][key]).correlation)
                            for a, b in ((0, 1), (0, 2), (1, 2))]
            row["stability_pairwise_spearman_mean"] = float(np.mean(correlations))
            row["stability_pairwise_spearman_std"] = float(np.std(correlations, ddof=1))
            aggregate.append(row)
    table(aggregate, OUT / "ablation_summary.csv")
    for score in scores:
        score.close()
    report = ["# Efficiency, Scalability, and Ablations", "",
              "This is a cached-score component benchmark, not end-to-end attribution or utility retraining.",
              "Timing: seed-0 clean cache, three process-isolated repetitions per scale. Ablations: three training-cache seeds, a fixed nested 10k subset of the 100k corpus, 10% selection.", "",
              "## Graph construction", "",
              "| Corpora | Exact GPU seconds | HNSW CPU seconds | ANN Recall@10 | Graph cache MiB |",
              "|---:|---:|---:|---:|---:|"]
    for n in (1000, 10000, 50000, 100000):
        rows = {r["stage"]: r for r in summary if r["n"] == n}
        matching = [p for p in payloads if p["rows"][0]["n"] == n]
        def fmt(stage):
            r = rows[stage]
            return "%.3f (%.3f)" % (r["seconds_mean"], r["seconds_std"])
        report.append("| %d | %s | %s | %.4f | %.3f |" % (
            n, fmt("exact_knn"), fmt("ann_knn"), np.mean([p["ann_recall10"] for p in matching]),
            matching[0]["graph_score_cache_bytes"] / MIB))
    report += ["", "Graph cache is uncompressed array payload (neighbors, semantic/gated weights, confidence, two scores), excluding embeddings, gradients and the ANN index.",
               "", "## All mechanism diagnostics", "",
               "Mean (sample SD), three cache seeds. AUROC uses negative contribution to detect corrupted examples. Pair affinity and coverage use the SAME semantic k=10 reference graph within each seed.", "",
               "| Variant | Order | Noise AUROC | Selected noise | Pair affinity | Coverage | Rank stability |",
               "|---|---:|---:|---:|---:|---:|---:|"]
    for row in aggregate:
        cells = ["%.5f (%.5f)" % (row[m + "_mean"], row[m + "_std"])
                 for m in ("noise_auroc", "selected_noise_fraction", "reference_pair_affinity", "reference_coverage")]
        report.append("| %s | %d | %s | %.5f |" % (row["variant"], row["order"], " | ".join(cells), row["stability_pairwise_spearman_mean"]))
    report += ["", "## Interpretation limits", "",
               "- Exact uses GPU float32 cosine, HNSW uses CPU normalized-L2 (M=32, efConstruction=80, efSearch=64, four threads). This is a deployment comparison, not a same-device algorithm speed comparison. HNSW insertion can be nondeterministic with multiple threads; all recall measurements are retained.",
               "- RSS includes the Python framework and loaded arrays. Five-millisecond sampling can miss short peaks; allocator retention can produce zero RSS deltas. CUDA measures PyTorch allocated bytes, not all device memory. Timing SD is not scientific seed variation.",
               "- The replay is a random-order pass over final cached scores, not the saved original training trajectory. Seconds divided by ceil(N/8) is amortized replay-batch overhead, NOT measured online graph-update latency.",
               "- No full-training wall time, gradient-cache generation cost, validation loss or accuracy was measured. This experiment cannot establish end-to-end superiority or recover the paper's exact dynamic method.",
               "- The 10k ablation corpus is a subset of the 100k-trained caches; it is not a newly trained 10k model. Gradient embedding is a gradient-space graph, not a second semantic encoder.",
               "- First-order no_pair is intentionally identical to full. Removing history removes the first-order correction entirely. Alpha zero disables correction but retains diversity selection. Zero SD is retained when genuine.",
               "- Pairwise rank-stability correlations share seeds and are dependent; they are descriptive, not independent confidence intervals. No hyperparameters are selected using these outcomes."]
    (OUT / "report.md").write_text("\n".join(report) + "\n")
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    colors = ["#247a60", "#b64b56", "#517ab4"]
    for stage, label, color in zip(("exact_knn", "ann_knn", "pointwise_replay"),
                                    ("Exact GPU KNN", "HNSW CPU KNN", "Pointwise score replay"), colors):
        group = [r for r in summary if r["stage"] == stage]
        axes[0].errorbar([r["n"] for r in group], [r["seconds_mean"] for r in group],
                         yerr=[r["seconds_std"] for r in group], label=label, color=color, marker="o", capsize=3)
    axes[0].set(xscale="log", yscale="log", xlabel="Corpora", ylabel="Component wall time (s)")
    axes[0].legend(fontsize=8)
    variants = ["full", "no_gate", "no_historical", "no_scale"]
    x = np.arange(len(variants))
    for order, offset, color in ((1, -.16, colors[0]), (2, .16, colors[1])):
        group = [next(r for r in aggregate if r["variant"] == v and r["order"] == order) for v in variants]
        for ax, metric in ((axes[1], "noise_auroc"), (axes[2], "reference_coverage")):
            ax.bar(x + offset, [r[metric + "_mean"] for r in group], .3,
                   yerr=[r[metric + "_std"] for r in group], capsize=3, color=color, label="Order %d" % order)
            ax.set_xticks(x)
            ax.set_xticklabels(["Full", "No gate", "No history", "No scale"], rotation=20)
    axes[1].set_ylabel("Noise AUROC")
    axes[2].set_ylabel("Reference graph coverage")
    axes[1].legend(fontsize=8)
    for ax in axes:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUT / ("efficiency_ablation." + suffix), dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["all", "bench", "ablate", "test", "summarize"])
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--rep", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(4)
    faiss.omp_set_num_threads(4)
    OUT.mkdir(exist_ok=True)
    (OUT / "timing").mkdir(exist_ok=True)
    (OUT / "ablation").mkdir(exist_ok=True)
    if args.mode == "test":
        tests()
    elif args.mode == "bench":
        benchmark(args.n, args.rep)
    elif args.mode == "ablate":
        ablation(args.seed)
    elif args.mode == "summarize":
        summarize()
    else:
        tests()
        dump(OUT / "provenance.json", dict(
            python=sys.version, numpy=np.__version__, torch=torch.__version__, faiss=faiss.__version__,
            gpu=torch.cuda.get_device_name(DEVICE), device=str(DEVICE), cpu_threads=4,
            source_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (Path(__file__).resolve(), ROOT / "pile_main_100k/common.py", ROOT / "pile_main_100k/graph_utils.py")},
            input_caches=[dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size,
                               mtime_ns=p.stat().st_mtime_ns) for p in sorted(CACHE.glob("*_seed*.npz"))],
            subset_seed=2027, sizes=[1000, 10000, 50000, 100000], cache_seeds=[0, 1, 2],
            timing_repetitions=3, batch_size=8, budget=.1, configurations=configurations()))
        for n in (1000, 10000, 50000, 100000):
            for rep in range(3):
                target = OUT / "timing" / ("n%d_rep%d.json" % (n, rep))
                if not target.exists():
                    subprocess.run([sys.executable, str(Path(__file__)), "bench", "--n", str(n), "--rep", str(rep)], check=True)
        for seed in range(3):
            if not (OUT / "ablation" / ("scores_seed%d.npz" % seed)).exists():
                subprocess.run([sys.executable, str(Path(__file__)), "ablate", "--seed", str(seed)], check=True)
        summarize()


if __name__ == "__main__":
    main()
