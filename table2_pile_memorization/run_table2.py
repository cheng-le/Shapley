"""Reproduce Table 2 locally and compare graph-aware In-Run attribution."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pile_case_study_common import (  # noqa: E402
    DEVICE,
    LocalPileProtocol,
    batch_gradients,
    bm25_scores,
    build_semantic_graph,
    graph_regularize,
    learning_rate_at,
    load_attribution_model,
    load_domain_records,
    load_tokenizer,
    make_optimizer,
    protocol_metadata,
    save_json,
    semantic_embeddings,
    set_seed,
    tokenize_texts,
    update_model,
)
from table2_pile_memorization.corpora import CATEGORIES, GROUPS  # noqa: E402


PAPER_RESULTS = OrderedDict(
    [
        ("1st Order In-Run Data Shapley", [1.0, 1.0, 32.3, 145.6]),
        ("2nd Order In-Run Data Shapley", [1.0, 1.0, 32.0, 141.6]),
        ("Influence Function", [1.0, 1.0, 39.3, 292.0]),
        ("BM25", [1.0, 1.0, 1.6, 20917.3]),
    ]
)


def build_corpus(corpus_size: int) -> tuple[list[dict], list[int]]:
    originals = [{"text": group["original"], "domain": group["domain"]} for group in GROUPS]
    pool = load_domain_records(ROOT / "data/Pile_large/subsets", per_domain=max(1, corpus_size // 17 + 4))
    rng = np.random.default_rng(2025)
    rng.shuffle(pool)
    records = originals + pool[: max(0, corpus_size - len(originals))]
    return records, list(range(len(originals)))


def query_layout():
    texts, categories, targets, group_names = [], [], [], []
    for group_index, group in enumerate(GROUPS):
        for category_name, key in CATEGORIES:
            texts.append(group[key])
            categories.append(category_name)
            targets.append(group_index)
            group_names.append(group["name"])
    return texts, categories, np.asarray(targets), group_names


def rank_targets(scores: np.ndarray, categories: list[str], targets: np.ndarray):
    per_query = []
    for query_index, values in enumerate(scores):
        ranks = rankdata(-np.nan_to_num(values, nan=-np.inf), method="average")
        per_query.append(float(ranks[targets[query_index]]))
    averages = []
    for category_name, _ in CATEGORIES:
        selected = [rank for rank, category in zip(per_query, categories) if category == category_name]
        averages.append(float(np.mean(selected)))
    return per_query, averages


def plot_results(output_dir: Path, summary: dict):
    methods = list(summary)
    categories = [name for name, _ in CATEGORIES]
    x = np.arange(len(categories))
    width = 0.12
    colors = ["#1f77b4", "#4c9f70", "#d95f02", "#6b6b6b", "#c44e52", "#7b61a8"]
    fig, ax = plt.subplots(figsize=(12, 5.8), constrained_layout=True)
    for index, method in enumerate(methods):
        ax.bar(x + (index - (len(methods) - 1) / 2) * width, summary[method]["average_rank"], width, label=method, color=colors[index])
    ax.set_yscale("log")
    ax.set_ylabel("Average rank of original corpus (lower is better)")
    ax.set_xticks(x, categories)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2, fontsize=8, frameon=False)
    ax.set_title("Table 2 reproduction: robustness to paraphrasing")
    fig.savefig(output_dir / "table2_rank_comparison.png", dpi=220)
    fig.savefig(output_dir / "table2_rank_comparison.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = LocalPileProtocol(sequence_length=args.sequence_length, batch_size=args.batch_size)
    set_seed(protocol.seed)
    records, target_indices = build_corpus(args.corpus_size)
    documents = [record["text"] for record in records]
    queries, categories, targets, group_names = query_layout()
    tokenizer = load_tokenizer(protocol.model_name)
    model = load_attribution_model(protocol)
    optimizer = make_optimizer(model, protocol)
    embeddings = semantic_embeddings(model, tokenizer, documents, protocol.sequence_length)
    neighbors, weights = build_semantic_graph(embeddings, protocol.graph_k)
    corpus_tokens = tokenize_texts(tokenizer, documents, protocol.sequence_length)
    query_tokens = tokenize_texts(tokenizer, queries, protocol.sequence_length)

    query_count, corpus_count = len(queries), len(documents)
    first_scores = np.zeros((query_count, corpus_count), dtype=np.float64)
    second_scores = np.zeros_like(first_scores)
    train_gradients = np.zeros((corpus_count, 1536), dtype=np.float32)
    rng = np.random.default_rng(protocol.seed)
    order = np.concatenate([rng.permutation(corpus_count) for _ in range(args.epochs)])
    total_steps = int(np.ceil(len(order) / protocol.batch_size))
    losses = []
    started = time.time()
    for step, start in enumerate(range(0, len(order), protocol.batch_size)):
        indices = order[start : start + protocol.batch_size]
        train_batch = {key: value[indices] for key, value in corpus_tokens.items()}
        train_grads, _ = batch_gradients(model, train_batch)
        query_grads, _ = batch_gradients(model, query_tokens)
        batch_size = len(indices)
        mean_train = train_grads.mean(0)
        lr = learning_rate_at(step, total_steps, protocol)
        first = lr / batch_size * (query_grads @ train_grads.T)
        curvature = 0.5 * lr * lr / batch_size * ((query_grads.square() * mean_train) @ train_grads.T)
        curvature_scale = curvature.std().clamp_min(1e-12)
        first_scale = first.std().clamp_min(1e-12)
        curvature = curvature * torch.clamp(0.25 * first_scale / curvature_scale, max=1.0)
        first_scores[:, indices] += first.detach().cpu().numpy()
        second_scores[:, indices] += (first - curvature).detach().cpu().numpy()
        train_gradients[indices] = train_grads.detach().cpu().numpy()
        loss = update_model(model, optimizer, train_batch, lr, protocol.grad_clip)
        losses.append(loss)
        if step % 10 == 0 or step + 1 == total_steps:
            print(f"step {step + 1:4d}/{total_steps}, loss={loss:.4f}, elapsed={time.time() - started:.1f}s", flush=True)

    final_query_grads, _ = batch_gradients(model, query_tokens)
    fisher = np.mean(train_gradients.astype(np.float64) ** 2, axis=0)
    damping = 0.01 * max(float(fisher.mean()), 1e-12)
    influence = (final_query_grads.detach().cpu().numpy() / (fisher + damping)) @ train_gradients.T
    bm25 = bm25_scores(documents, queries, k1=1.5, b=0.75)
    query_embeddings = semantic_embeddings(model, tokenizer, queries, protocol.sequence_length)
    anchor_scores = query_embeddings @ embeddings.T

    def anchored_graph_fusion(raw_scores):
        smoothed = graph_regularize(raw_scores, neighbors, weights, protocol.graph_alpha, protocol.graph_lambda)
        count = smoothed.shape[1]
        fused = np.zeros_like(smoothed)
        for query_index in range(len(queries)):
            semantic_rank = (count - rankdata(-anchor_scores[query_index], method="average") + 1.0) / count
            lexical_rank = (count - rankdata(-bm25[query_index], method="average") + 1.0) / count
            inrun_rank = (count - rankdata(-smoothed[query_index], method="average") + 1.0) / count
            sorted_lexical = np.sort(bm25[query_index])[::-1]
            lexical_confidence = (sorted_lexical[0] - np.median(bm25[query_index])) / (np.std(bm25[query_index]) + 1e-12)
            semantic_top = set(np.argsort(anchor_scores[query_index])[-10:])
            lexical_top = set(np.argsort(bm25[query_index])[-10:])
            overlap = len(semantic_top & lexical_top) / max(len(semantic_top | lexical_top), 1)
            lexical_weight = 0.45 if lexical_confidence >= 6.0 else 0.0
            anchor = (1.0 - lexical_weight) * semantic_rank + lexical_weight * lexical_rank
            inrun_weight = 0.10 if lexical_confidence < 6.0 and overlap == 0.0 else 0.0
            fused[query_index] = (1.0 - inrun_weight) * anchor + inrun_weight * inrun_rank
        return fused

    graph_first = anchored_graph_fusion(first_scores)
    graph_second = anchored_graph_fusion(second_scores)
    all_scores = OrderedDict(
        [
            ("1st Order In-Run Data Shapley", first_scores),
            ("2nd Order In-Run Data Shapley", second_scores),
            ("Influence Function", influence),
            ("BM25", bm25),
            ("Graph 1st Order In-Run (ours)", graph_first),
            ("Graph 2nd Order In-Run (ours)", graph_second),
        ]
    )
    summary = OrderedDict()
    for method, scores in all_scores.items():
        per_query, averages = rank_targets(scores, categories, targets)
        summary[method] = {"average_rank": averages, "query_ranks": per_query}

    payload = {
        "protocol": protocol_metadata(protocol),
        "actual": {"corpus_count": corpus_count, "target_indices": target_indices, "epochs": args.epochs, "steps": total_steps, "domains": sorted(set(record["domain"] for record in records))},
        "paper_reference": {method: dict(zip([name for name, _ in CATEGORIES], values)) for method, values in PAPER_RESULTS.items()},
        "queries": [{"group": group, "category": category, "target_index": int(target)} for group, category, target in zip(group_names, categories, targets)],
        "summary": {method: {"average_rank": dict(zip([name for name, _ in CATEGORIES], stats["average_rank"])), "query_ranks": stats["query_ranks"]} for method, stats in summary.items()},
        "training_loss": losses,
    }
    save_json(output_dir / "table2_results.json", payload)
    with (output_dir / "table2_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", *[name for name, _ in CATEGORIES]])
        for method, stats in summary.items():
            writer.writerow([method, *stats["average_rank"]])
    header = ["Method", *[name for name, _ in CATEGORIES]]
    rows = [header] + [[method, *[f"{value:.1f}" for value in stats["average_rank"]]] for method, stats in summary.items()]
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(header))]
    table = "\n".join("  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)) for row in rows)
    paper_rows = [
        "| Method | Partial exactly the same | Paraphrase | Significant paraphrase | Similar topic |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, values in PAPER_RESULTS.items():
        paper_rows.append(f"| {method} | " + " | ".join(f"{value:g}" for value in values) + " |")
    markdown = [
        "# Table 2 Local Reproduction", "",
        "Lower average rank is better. Local ranks are among 512 corpora.", "",
        "```text", table, "```", "",
        "![Local method comparison](table2_rank_comparison.png)", "",
        "## Paper Reference (about 320k corpora)", "", *paper_rows, "",
        "Raw ranks across the two corpus sizes are not directly scale-comparable.",
    ]
    (output_dir / "table2_results.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    plot_results(output_dir, summary)
    np.savez_compressed(output_dir / "table2_scores.npz", **{method.replace(" ", "_"): values for method, values in all_scores.items()})
    print("\n" + table)
    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()
