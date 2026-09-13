"""Pile Appendix E, using only the genuine Hessian-free 1k online experiment."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import textwrap

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "pile_main_100k"))
import numpy as np
import torch
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from common import model, example_losses, trainable, tokenizer, diversity_select

SOURCE = ROOT / "efficiency_scalability_ablations/second_order_online"
OUT = HERE / "results"
METHODS = ("inrun2", "adam2", "graph", "adam1")
NAMES = {"inrun2": "In-Run second", "adam2": "Adam second", "graph": "Graph no-HVP", "adam1": "Adam first",
         "block_noise": "Noise-source blocking (oracle)", "graph_diverse": "Graph diverse"}


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def table(name, rows):
    with (OUT / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def pct(scores):
    return 1 - (rankdata(-scores, method="average") - 1) / (len(scores) - 1)


def messages(raw, order, nn, ww, noise, block=False):
    history = np.zeros_like(raw)
    scores = np.zeros_like(raw)
    allocation = np.zeros((4, len(raw)), dtype=np.float32)
    steps = []
    for number, start in enumerate(range(0, len(raw), 8)):
        ids = order[start:start + 8]
        values = raw[ids]
        local = np.zeros_like(raw)
        local[ids] = values
        neighbors, weights = nn[ids], ww[ids]
        parts = []
        for source in (history, local):
            clean = (weights * source[neighbors] * (1 - noise[neighbors])).sum(1)
            corrupt = (weights * source[neighbors] * noise[neighbors]).sum(1)
            parts.extend([clean, np.zeros_like(corrupt) if block else corrupt])
        components = []
        for offset, strength in ((0, .2), (2, .1)):
            term = parts[offset] + parts[offset + 1]
            a, b = float(values.std()), float(term.std())
            factor = strength * a / (b + 1e-12) if min(a, b) >= 1e-12 else 0.
            components.extend([(p - float(p.mean())) * factor for p in parts[offset:offset + 2]])
        scores[ids] = values + sum(components)
        for i, c in enumerate(components):
            allocation[i, ids] = c
        history[ids] += values
        steps.append(dict(step=number, clean_history_l1=float(np.abs(parts[0]).sum()),
            noise_history_l1=float(np.abs(parts[1]).sum()), clean_pair_l1=float(np.abs(parts[2]).sum()),
            noise_pair_l1=float(np.abs(parts[3]).sum())))
    return scores, allocation, steps


def worker(seed):
    torch.set_num_threads(2)
    device = torch.device("cuda:%d" % seed)
    net = model(device)
    net.eval()
    with np.load(SOURCE / "data" / ("noise_seed%d.npz" % seed)) as z:
        ids, mask = z["ids"], z["mask"]
    def batch(start, stop):
        return {"input_ids": torch.from_numpy(ids[start:stop]).long().to(device),
                "attention_mask": torch.from_numpy(mask[start:stop]).long().to(device)}
    features, gradients = [], []
    for start in range(0, 1000, 8):
        b = batch(start, start + 8)
        with torch.no_grad():
            h = net.transformer(**b).last_hidden_state.float()
            m = b["attention_mask"].unsqueeze(-1)
            pooled = (h * m).sum(1) / m.sum(1).clamp_min(1)
            features.append(torch.nn.functional.normalize(pooled, dim=1).cpu().numpy())
        losses = example_losses(net, b)
        for pos, loss in enumerate(losses):
            g = torch.autograd.grad(loss, trainable(net), retain_graph=pos + 1 < len(losses))
            gradients.append(torch.cat([p.reshape(-1) for p in g]).detach().cpu().numpy())
    with np.load(SOURCE / "data/validation.npz") as z:
        b = {"input_ids": torch.from_numpy(z["ids"][:16]).long().to(device),
             "attention_mask": torch.from_numpy(z["mask"][:16]).long().to(device)}
    with torch.no_grad():
        h = net.transformer(**b).last_hidden_state.float()
        m = b["attention_mask"].unsqueeze(-1)
        q = torch.nn.functional.normalize((h * m).sum(1) / m.sum(1).clamp_min(1), dim=1).cpu().numpy()
    np.savez_compressed(OUT / ("initial_diagnostics_seed%d.npz" % seed),
                        features=np.concatenate(features), gradients=np.stack(gradients), query_features=q)
    print("Initial-state diagnostics", seed, "complete", flush=True)


def analyze():
    manifest = json.loads((SOURCE / "data/manifest.json").read_text())
    source_ids = np.array(manifest["train_indices"])
    wanted = set(map(int, source_ids))
    domains_by_id = {}
    with (ROOT / "data/Pile_large/train.jsonl").open() as f:
        for i, line in enumerate(f):
            if i in wanted:
                row = json.loads(line)
                domains_by_id[i] = row.get("meta", {}).get("pile_set_name", "Unknown")
    domains = np.array([domains_by_id[int(i)] for i in source_ids])
    nodes, edges, contamination, distributions, buckets, effects, all_steps, stability, matrix_rows = [], [], [], [], [], [], [], [], []
    figures, hashes, cases = {}, {}, []
    max_error = 0.
    for scenario in ("clean", "noise"):
        vectors, labels = [], []
        for seed in range(3):
            for method in METHODS:
                path = SOURCE / "results/cells" / ("%s_seed%d_%s.npz" % (scenario, seed, method))
                hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
                with np.load(path) as z:
                    scores = z["scores"]
                vectors.append(scores)
                labels.append(method + "/s%d" % seed)
        matrix = spearmanr(np.stack(vectors), axis=1).correlation
        assert np.allclose(matrix, matrix.T) and np.allclose(np.diag(matrix), 1)
        np.savez_compressed(OUT / (scenario + "_rank_matrix.npz"), matrix=matrix, labels=np.array(labels))
        for i, left in enumerate(labels):
            for j, right in enumerate(labels):
                matrix_rows.append(dict(scenario=scenario, left=left, right=right, spearman=float(matrix[i, j])))
        for method in METHODS:
            for a, b in ((0, 1), (0, 2), (1, 2)):
                stability.append(dict(scenario=scenario, method=method, seed_a=a, seed_b=b,
                    spearman=float(matrix[labels.index(method + "/s%d" % a), labels.index(method + "/s%d" % b)])))
    for seed in range(3):
        data_path = SOURCE / "data" / ("noise_seed%d.npz" % seed)
        hashes[str(data_path.relative_to(ROOT))] = hashlib.sha256(data_path.read_bytes()).hexdigest()
        with np.load(data_path) as data:
            noise, token_ids, token_mask = data["labels"], data["ids"], data["mask"]
        assert int(noise.sum()) == 100
        scores = {}
        for method in METHODS:
            path = SOURCE / "results/cells" / ("noise_seed%d_%s.npz" % (seed, method))
            with np.load(path) as z:
                scores[method] = z["scores"]
                if method == "graph":
                    nn, ww, order, first = z["neighbors"], z["weights"], z["order"], z["first"]
            audit = json.loads(path.with_suffix(".json").read_text())["audit"]
            assert audit["hvp_calls"] == (0 if method in ("graph", "adam1") else 125)
        np.testing.assert_array_equal(first, scores["adam1"])
        factual, allocations, steps = messages(first, order, nn, ww, noise)
        np.testing.assert_allclose(factual, scores["graph"], rtol=5e-5, atol=1e-9)
        np.testing.assert_allclose(factual - first, allocations.sum(0), rtol=5e-5, atol=1e-9)
        max_error = max(max_error, float(np.max(np.abs(factual - scores["graph"]))))
        blocked = messages(first, order, nn, ww, noise, True)[0]
        np.testing.assert_allclose(messages(first, order, nn, ww, np.zeros(1000, dtype=np.int8), True)[0], factual, rtol=5e-5, atol=1e-9)
        np.testing.assert_array_equal(messages(first, order, nn, ww, np.ones(1000, dtype=np.int8), True)[0], first)
        all_steps.extend(dict(seed=seed, **r) for r in steps)
        np.savez_compressed(OUT / ("propagation_seed%d.npz" % seed), factual=factual, blocked=blocked, allocations=allocations)
        with np.load(OUT / ("initial_diagnostics_seed%d.npz" % seed)) as z:
            features, grad, queries = z["features"], z["gradients"], z["query_features"]
        norms = np.linalg.norm(grad, axis=1).clip(1e-8)
        cosine = np.empty(nn.shape, dtype=np.float32)
        for start in range(0, 1000, 32):
            cosine[start:start + 32] = (grad[start:start + 32, None] * grad[nn[start:start + 32]]).sum(2) / (norms[start:start + 32, None] * norms[nn[start:start + 32]])
        gradient_agreement = (ww * cosine).sum(1)
        density = (features[:, None] * features[nn]).sum(2).mean(1)
        adjacency = np.zeros((1000, 1000), dtype=np.float32)
        adjacency[np.arange(1000)[:, None], nn] = ww
        degree = ((adjacency + adjacency.T) > 0).sum(1)
        domain_agreement = (ww * (domains[:, None] == domains[nn])).sum(1)
        incoming_noise = (ww * noise[nn]).sum(1)
        lift = pct(scores["graph"]) - pct(first)
        selected_mask = np.zeros(1000, dtype=bool)
        selected_mask[np.argsort(-scores["graph"], kind="stable")[:200]] = True
        for i in range(1000):
            nodes.append(dict(seed=seed, local_id=i, source_id=int(source_ids[i]), original_domain=domains[i], corrupted=int(noise[i]),
                raw_score=float(first[i]), graph_score=float(scores["graph"][i]), percentile_lift=float(lift[i]),
                blocked_percentile_delta=float(pct(blocked)[i] - pct(scores["graph"])[i]),
                domain_agreement=float(domain_agreement[i]), initial_gradient_agreement=float(gradient_agreement[i]),
                degree=int(degree[i]), raw_cosine_density=float(density[i]), incoming_noise_weight=float(incoming_noise[i]),
                clean_history=float(allocations[0, i]), noise_history=float(allocations[1, i]),
                clean_pair=float(allocations[2, i]), noise_pair=float(allocations[3, i])))
            for j, source in enumerate(nn[i]):
                edges.append(dict(seed=seed, recipient=i, source=int(source), recipient_corrupted=int(noise[i]),
                    source_corrupted=int(noise[source]), weight=float(ww[i, j]), initial_gradient_cosine=float(cosine[i, j])))
        for method, values in scores.items():
            z = (values - values.mean()) / max(float(values.std()), 1e-12)
            distributions.extend(dict(seed=seed, method=method, local_id=i, corrupted=int(noise[i]),
                score=float(values[i]), within_seed_z=float(z[i])) for i in range(1000))
        for method, values in dict(scores, block_noise=blocked, graph_diverse=scores["graph"]).items():
            for budget in (.01, .02, .05, .1, .2, .4):
                k = round(1000 * budget)
                chosen = (diversity_select(values, nn, ww, k, lambda_fraction=.05)[0] if method == "graph_diverse"
                          else np.argsort(-values, kind="stable")[:k])
                contamination.append(dict(seed=seed, method=method, budget=budget, count=k,
                    corrupted_selected=int(noise[chosen].sum()), contamination=float(noise[chosen].mean()),
                    noise_auroc=float(roc_auc_score(noise, -values))))
        for variable, values in (("domain_agreement", domain_agreement), ("initial_gradient_agreement", gradient_agreement),
                                 ("degree", degree), ("density", density)):
            cuts = np.unique(np.quantile(values, np.linspace(0, 1, 6)))
            labels = np.digitize(values, cuts[1:-1], right=True)
            for bucket in np.unique(labels):
                mask = labels == bucket
                selected = mask & selected_mask
                buckets.append(dict(seed=seed, variable=variable, bucket=int(bucket), lower=float(values[mask].min()),
                    upper=float(values[mask].max()), count=int(mask.sum()), corrupted_count=int(noise[mask].sum()),
                    noise_fraction=float(noise[mask].mean()), selected_count=int(selected.sum()),
                    selected_noise_fraction=float(noise[selected].mean()) if selected.any() else None,
                    mean_percentile_lift=float(lift[mask].mean()),
                    noise_auroc=float(roc_auc_score(noise[mask], -scores["graph"][mask])) if len(np.unique(noise[mask])) == 2 else None))
        for status in (0, 1):
            mask = noise == status
            effects.append(dict(seed=seed, corrupted=status, count=int(mask.sum()),
                promoted_fraction=float((lift[mask] > 0).mean()), mean_percentile_lift=float(lift[mask].mean()),
                blocked_percentile_delta=float((pct(blocked)[mask] - pct(scores["graph"])[mask]).mean()),
                noise_incoming_weight=float(incoming_noise[mask].mean())))
        figures[seed] = (first, scores["graph"], noise)
        if seed == 0:
            tok = tokenizer()
            with np.load(SOURCE / "data/clean_seed0.npz") as z:
                clean_tokens, clean_masks = z["ids"], z["mask"]
            with np.load(SOURCE / "data/validation.npz") as z:
                query_tokens, query_masks = z["ids"][:16], z["mask"][:16]
            def excerpt(ids, masks, i):
                return tok.decode(ids[i][masks[i].astype(bool)]).strip()
            for status, op, title in ((1, np.argmin, "Corrupt: largest demotion"), (1, np.argmax, "Corrupt: largest promotion"),
                                      (0, np.argmax, "Clean: largest promotion"), (0, np.argmin, "Clean: largest demotion")):
                candidates = np.flatnonzero(noise == status)
                i = int(candidates[op(lift[candidates])])
                q = int(np.argmax(queries @ features[i]))
                neighbors = np.argsort(-ww[i], kind="stable")[:5]
                cases.append(dict(title=title, seed=0, local_id=i, source_id=int(source_ids[i]),
                    original_domain=domains[i], corrupted=status, raw_rank=int(rankdata(-first, method="min")[i]),
                    graph_rank=int(rankdata(-scores["graph"], method="min")[i]), percentile_lift=float(lift[i]),
                    actual_model_input=excerpt(token_ids, token_mask, i), clean_model_input=excerpt(clean_tokens, clean_masks, i),
                    closest_utility_query=dict(local_id=q, source_validation_id=manifest["utility_val_indices"][q],
                        cosine=float(queries[q] @ features[i]), text=excerpt(query_tokens, query_masks, q)),
                    neighbors=[dict(local_id=int(nn[i, p]), source_id=int(source_ids[nn[i, p]]), original_domain=domains[nn[i, p]],
                        corrupted=int(noise[nn[i, p]]), weight=float(ww[i, p]), text=excerpt(token_ids, token_mask, int(nn[i, p]))) for p in neighbors]))
        print("Pile seed", seed, "analyzed", flush=True)
    for name, rows in (("nodes", nodes), ("edges", edges), ("contamination", contamination), ("score_distribution", distributions),
                       ("buckets", buckets), ("propagation_effects", effects), ("step_messages", all_steps),
                       ("stability_pairs", stability), ("rank_correlations", matrix_rows)):
        table(name + ".csv", rows)
    dump(OUT / "cases.json", cases)
    case_text = ["# Pile Neighborhood Cases", "", "Fixed seed 0; extrema of percentile change, not representative frequencies. Snippets are the actual <=128-token model inputs. Query is the closest of the 16 actual attribution utility queries by initial GPT-2 cosine, not a proven causal explanation.", ""]
    for c in cases:
        case_text += ["## " + c["title"], "", "Source %d; original domain %s; raw rank %d -> Graph rank %d." % (c["source_id"], c["original_domain"], c["raw_rank"], c["graph_rank"]),
            "", "Actual input:", "", "```text", c["actual_model_input"], "```", "", "Clean counterpart:", "", "```text", c["clean_model_input"], "```", "",
            "Closest utility query (validation source %d):" % c["closest_utility_query"]["source_validation_id"], "", "```text", c["closest_utility_query"]["text"], "```", ""]
        for n in c["neighbors"]:
            case_text += ["Neighbor %d; domain %s; corrupted=%d; weight=%.5f:" % (n["source_id"], n["original_domain"], n["corrupted"], n["weight"]), "", "```text", n["text"], "```", ""]
    (OUT / "cases.md").write_text("\n".join(case_text) + "\n")
    plots(figures, contamination, buckets, all_steps)
    dump(OUT / "provenance.json", dict(source_hashes=hashes, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source=str(SOURCE.relative_to(ROOT)), corpus_size=1000, graph="semantic, no gate, no HVP",
        diagnostic_gradients="initial pretrained GPT-2 final LayerNorm, eval mode; never fed into scores"))
    dump(OUT / "validation.json", dict(status="passed", seeds=3, nodes=len(nodes), edges=len(edges),
        distributions=len(distributions), contamination_cells=len(contamination), rank_cells=len(matrix_rows),
        max_replay_absolute_error=max_error, allocation_reconstruction=True, counterfactual_endpoints=True))


def plots(figures, contamination, buckets, steps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "svg.fonttype": "none"})
    def save(fig, name):
        fig.tight_layout()
        for suffix in ("png", "pdf", "svg"):
            fig.savefig(OUT / (name + "." + suffix), dpi=180)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, scenario in zip(axes, ("clean", "noise")):
        with np.load(OUT / (scenario + "_rank_matrix.npz")) as z:
            matrix, labels = z["matrix"], z["labels"]
        ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set(xticks=range(12), yticks=range(12), xticklabels=labels, yticklabels=labels, title=scenario + " / Spearman")
        ax.tick_params(axis="x", labelrotation=90, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
    save(fig, "rank_matrices")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for seed, ax in enumerate(axes):
        raw, graph, noise = figures[seed]
        vals = []
        for scores in (raw, graph):
            z = (scores - scores.mean()) / max(float(scores.std()), 1e-12)
            vals.extend([z[noise == 0], z[noise == 1]])
        ax.violinplot(vals, showmedians=True, showextrema=False)
        ax.set(xticks=range(1, 5), xticklabels=["Raw clean", "Raw noise", "Graph clean", "Graph noise"], title="Seed %d" % seed, ylabel="Within-seed score z")
        ax.tick_params(axis="x", labelrotation=30)
    save(fig, "per_seed_distributions")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method in NAMES:
        budgets = [.01, .02, .05, .1, .2, .4]
        vals = [[r["contamination"] for r in contamination if r["method"] == method and r["budget"] == b] for b in budgets]
        ax.errorbar(np.array(budgets) * 100, np.mean(vals, axis=1), yerr=np.std(vals, axis=1, ddof=1),
                    marker="o", capsize=2, linewidth=1, label=NAMES[method])
    ax.axhline(.1, color="gray", linestyle="--")
    ax.set(xlabel="Retention (%)", ylabel="Selected corruption fraction")
    ax.legend(fontsize=7)
    save(fig, "topk_contamination")
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5))
    for ax, variable in zip(axes, ("domain_agreement", "initial_gradient_agreement", "degree", "density")):
        for seed in range(3):
            group = [r for r in buckets if r["variable"] == variable and r["seed"] == seed]
            ax.plot([r["bucket"] + 1 for r in group], [r["noise_fraction"] for r in group], "o-", label="seed%d" % seed)
        ax.set(title=variable.replace("_", " "), xlabel="Quantile bin (ties merged)", ylabel="Corruption fraction")
    axes[0].legend(fontsize=6)
    save(fig, "neighborhood_buckets")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, term in zip(axes, ("history", "pair")):
        for label in ("clean", "noise"):
            values = [[r[label + "_" + term + "_l1"] for r in steps if r["step"] == i] for i in range(125)]
            ax.plot(range(125), np.mean(values, axis=1), label=label + " sources")
        ax.set(title=term, xlabel="Actual trajectory step", ylabel="Absolute raw message mass")
        ax.legend()
    save(fig, "noise_messages")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        worker(int(sys.argv[2]))
        return
    procs = []
    for seed in range(3):
        if not (OUT / ("initial_diagnostics_seed%d.npz" % seed)).exists():
            procs.append(subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "worker", str(seed)]))
    codes = [p.wait() for p in procs]
    if any(codes):
        raise RuntimeError("Initial diagnostic workers failed: %s" % codes)
    analyze()


if __name__ == "__main__":
    main()
