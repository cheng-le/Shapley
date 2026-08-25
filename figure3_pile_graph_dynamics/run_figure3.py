"""Track graph-aware second-order Pile domain contributions during GPT-2 training."""

from __future__ import annotations

import argparse
import csv
import sys
import textwrap
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pile_case_study_common import (  # noqa: E402
    LocalPileProtocol,
    batch_gradients,
    build_semantic_graph,
    graph_regularize,
    learning_rate_at,
    load_attribution_model,
    load_domain_records,
    load_jsonl_records,
    load_tokenizer,
    make_optimizer,
    protocol_metadata,
    save_json,
    semantic_embeddings,
    set_seed,
    tokenize_texts,
    update_model,
)


def domain_composition(scores, domains, seen):
    totals = {}
    for domain in sorted(set(domains)):
        mask = (domains == domain) & seen
        totals[domain] = float(scores[mask].sum()) if mask.any() else 0.0
    positive = {domain: max(value, 0.0) for domain, value in totals.items()}
    denominator = sum(positive.values())
    if denominator <= 0:
        return totals, {domain: 0.0 for domain in totals}
    return totals, {domain: 100.0 * value / denominator for domain, value in positive.items()}


def plot_dynamics(output_dir, history, domain_sizes, records, final_scores):
    domains = list(domain_sizes)
    steps = np.asarray([row["step"] for row in history])
    matrix = np.asarray([[row["composition"][domain] for row in history] for domain in domains])
    palette = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("Set3").colors)
    colors = palette[: len(domains)]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw={"width_ratios": [2.4, 1]}, constrained_layout=True)
    axes[0].stackplot(steps, matrix, labels=domains, colors=colors, linewidth=0.15)
    axes[0].set(xlabel="Number of gradient updates", ylabel="Domain contribution (%)", ylim=(0, 100))
    axes[0].set_title("Graph second-order contribution by Pile domain")
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].barh(domains, [domain_sizes[d] for d in domains], color=colors)
    axes[1].set_xlabel("Number of local training corpora")
    axes[1].set_title("Local domain size")
    axes[1].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc="center right", bbox_to_anchor=(1.13, 0.5), frameon=False, fontsize=8)
    fig.savefig(output_dir / "figure3_graph_domain_contribution.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "figure3_graph_domain_contribution.pdf", bbox_inches="tight")
    plt.close(fig)

    high = np.argsort(final_scores)[-3:][::-1]
    low = np.argsort(final_scores)[:3]
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), constrained_layout=True)
    for ax, indices, title, color in [
        (axes[0], high, "High-valued training corpora", "#cc6600"),
        (axes[1], low, "Low-valued training corpora", "#666666"),
    ]:
        ax.axis("off")
        ax.set_title(title, loc="left", color=color, fontsize=13, fontweight="bold")
        rows = []
        for rank, index in enumerate(indices, 1):
            snippet = textwrap.shorten(" ".join(records[index]["text"].split()), width=170, placeholder="...")
            rows.append(f"{rank}. [{records[index]['domain']}] score={final_scores[index]:.3e}\n{snippet}")
        ax.text(0.01, 0.92, "\n\n".join(rows), va="top", family="monospace", fontsize=9)
    fig.savefig(output_dir / "figure3_high_low_corpora.png", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-domain", type=int, default=32)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--record-every", type=int, default=5)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--math-validation-size", type=int, default=4)
    args = parser.parse_args()
    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = LocalPileProtocol(sequence_length=args.sequence_length, batch_size=args.batch_size)
    set_seed(protocol.seed)
    records = load_domain_records(ROOT / "data/Pile_large/subsets", args.per_domain)
    math_rows = load_jsonl_records(ROOT / "data/Pile_large/subsets/dm_mathematics.jsonl", args.per_domain + args.math_validation_size)
    validation = [row["text"] for row in math_rows[-args.math_validation_size :]]
    texts = [row["text"] for row in records]
    domains = np.asarray([row["domain"] for row in records])
    tokenizer = load_tokenizer(protocol.model_name)
    model = load_attribution_model(protocol)
    optimizer = make_optimizer(model, protocol)
    embeddings = semantic_embeddings(model, tokenizer, texts, protocol.sequence_length)
    neighbors, weights = build_semantic_graph(embeddings, protocol.graph_k)
    tokens = tokenize_texts(tokenizer, texts, protocol.sequence_length)
    validation_tokens = tokenize_texts(tokenizer, validation, protocol.sequence_length)

    corpus_count = len(records)
    raw_scores = np.zeros(corpus_count, dtype=np.float64)
    seen = np.zeros(corpus_count, dtype=bool)
    visits = np.zeros(corpus_count, dtype=np.int64)
    history = []
    training_loss = []
    rng = np.random.default_rng(protocol.seed)
    order = np.empty(0, dtype=np.int64)
    cursor = 0
    started = time.time()
    for step in range(args.steps):
        if cursor + protocol.batch_size > len(order):
            order = rng.permutation(corpus_count)
            cursor = 0
        indices = order[cursor : cursor + protocol.batch_size]
        cursor += protocol.batch_size
        batch = {key: value[indices] for key, value in tokens.items()}
        train_grads, _ = batch_gradients(model, batch)
        validation_grads, _ = batch_gradients(model, validation_tokens)
        validation_grad = validation_grads.mean(0)
        mean_train = train_grads.mean(0)
        lr = learning_rate_at(step, args.steps, protocol)
        first = lr / len(indices) * (train_grads @ validation_grad)
        curvature = 0.5 * lr * lr / len(indices) * (train_grads @ (validation_grad.square() * mean_train))
        if curvature.std() > 0:
            curvature = curvature * min(1.0, float(0.25 * first.std().clamp_min(1e-12) / curvature.std()))
        local_second = (first - curvature).detach().cpu().numpy()
        raw_scores[indices] += local_second
        seen[indices] = True
        visits[indices] += 1
        loss = update_model(model, optimizer, batch, lr, protocol.grad_clip)
        training_loss.append(loss)
        if step % args.record_every == 0 or step + 1 == args.steps:
            graph_scores = graph_regularize(raw_scores, neighbors, weights, protocol.graph_alpha, protocol.graph_lambda)
            totals, composition = domain_composition(graph_scores, domains, seen)
            history.append({"step": step + 1, "composition": composition, "domain_totals": totals, "loss": loss})
        if step % 25 == 0 or step + 1 == args.steps:
            print(f"step {step + 1:4d}/{args.steps}, loss={loss:.4f}, seen={seen.sum()}/{corpus_count}, elapsed={time.time() - started:.1f}s", flush=True)

    final_scores = graph_regularize(raw_scores, neighbors, weights, protocol.graph_alpha, protocol.graph_lambda)
    domain_sizes = dict(sorted(Counter(domains.tolist()).items()))
    payload = {
        "protocol": protocol_metadata(protocol),
        "paper_figure_target": {"attribution": "second-order In-Run Data Shapley", "validation": "math-related corpus", "tracked_steps": 10000, "normalization": "exclude domains with total value < 0, then normalize positive totals"},
        "actual": {"method": "Graph second-order In-Run (ours)", "corpus_count": corpus_count, "domain_count": len(domain_sizes), "domain_sizes": domain_sizes, "steps": args.steps, "record_every": args.record_every, "math_validation_corpora": len(validation)},
        "history": history,
        "final_scores": final_scores.tolist(),
        "visits": visits.tolist(),
        "training_loss": training_loss,
        "high_value": [{"index": int(i), "domain": records[i]["domain"], "score": float(final_scores[i]), "text": records[i]["text"][:500]} for i in np.argsort(final_scores)[-3:][::-1]],
        "low_value": [{"index": int(i), "domain": records[i]["domain"], "score": float(final_scores[i]), "text": records[i]["text"][:500]} for i in np.argsort(final_scores)[:3]],
    }
    save_json(output_dir / "figure3_results.json", payload)
    with (output_dir / "figure3_domain_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", *domain_sizes])
        for row in history:
            writer.writerow([row["step"], *[row["composition"][domain] for domain in domain_sizes]])
    plot_dynamics(output_dir, history, domain_sizes, records, final_scores)
    final_composition = history[-1]["composition"]
    rows = ["| Domain | Final contribution | Raw total |", "|---|---:|---:|"]
    for domain in sorted(domain_sizes, key=lambda name: final_composition[name], reverse=True):
        rows.append(f"| {domain} | {final_composition[domain]:.3f}% | {history[-1]['domain_totals'][domain]:+.3e} |")
    markdown = [
        "# Figure 3: Graph Domain Dynamics", "",
        "![Graph domain contribution](figure3_graph_domain_contribution.png)", "",
        "![High- and low-valued corpora](figure3_high_low_corpora.png)", "",
        "## Final Domain Composition", "", *rows, "",
        "Domains with negative total value are excluded before positive totals are normalized to 100%.",
    ]
    (output_dir / "figure3_results.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()
