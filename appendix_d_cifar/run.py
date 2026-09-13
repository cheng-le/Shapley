"""Appendix D: frozen-ResNet CIFAR graph ablations with real head retraining."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pile_main_100k"))
sys.path.insert(0, str(ROOT / "efficiency_scalability_ablations/vendor"))
import faiss
import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from torchvision.models import resnet18
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize
from common import confidence_gated_graph_weights, diversity_select

OUT = HERE / "results"
DATA = HERE / "data"


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def csv_write(path, rows):
    with path.open("w", newline="") as f:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def configs():
    base = dict(k=10, alpha=.2, lam=.05, gate=True, history=True, pair=True,
                scale=True, embedding="resnet128", sym="directed", ann=0, graph_seed=0)
    variants = []
    for k in (5, 10, 20):
        for a in (0., .2, .4):
            for lam in (0., .05, .1):
                name = "full" if (k, a, lam) == (10, .2, .05) else "k%d_a%g_l%g" % (k, a, lam)
                variants.append((name, dict(base, k=k, alpha=a, lam=lam)))
    for flag in ("gate", "history", "pair", "scale"):
        variants.append(("no_" + flag, dict(base, **{flag: False})))
    for embedding in ("resnet32", "pixels8"):
        variants.append((embedding, dict(base, embedding=embedding)))
    for sym in ("mean", "mutual"):
        variants.append(("sym_" + sym, dict(base, sym=sym)))
    for ef in (2, 8, 32, 128):
        variants.append(("ann%d" % ef, dict(base, ann=ef)))
    for seed in (1, 2):
        variants.append(("ann8_seed%d" % seed, dict(base, ann=8, graph_seed=seed)))
        variants.append(("exact_seed%d" % seed, dict(base, graph_seed=seed)))
    variants.append(("random", dict(base)))
    return variants


def prepare():
    DATA.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:3")
    transform = Compose([ToTensor(), Normalize([.485, .456, .406], [.229, .224, .225])])
    train, test = [CIFAR10(str(ROOT / "data/CIFAR10"), train=flag, download=False,
                          transform=transform) for flag in (True, False)]
    perm = np.random.default_rng(2027).permutation(len(train))
    validation, eligible = perm[:1000], perm[1000:]
    test_ids = np.random.default_rng(2028).permutation(len(test))[:2000]
    subsets = [np.random.default_rng(s).choice(eligible, 1000, replace=False) for s in range(3)]
    union = np.unique(np.concatenate(subsets + [validation]))
    checkpoint = DATA / "resnet18-f37072fd.pth"
    if not checkpoint.exists():
        checkpoint = Path(torch.hub.get_dir()) / "checkpoints/resnet18-f37072fd.pth"
    if not checkpoint.exists():
        raise FileNotFoundError("Local ImageNet checkpoint required: %s" % checkpoint)
    net = resnet18(weights=None)
    net.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    net.fc = torch.nn.Identity()
    net.to(device).eval()
    def extract(dataset, ids):
        collected = {k: [] for k in ("resnet128", "resnet32", "pixels8")}
        with torch.no_grad():
            for start in range(0, len(ids), 64):
                images = torch.stack([dataset[int(i)][0] for i in ids[start:start + 64]]).to(device)
                for size in (128, 32):
                    resized = F.interpolate(images, size=(size, size), mode="bilinear", align_corners=False)
                    collected["resnet%d" % size].append(F.normalize(net(resized), dim=1).cpu().numpy())
                pixels = F.adaptive_avg_pool2d(images, (8, 8)).flatten(1)
                collected["pixels8"].append(F.normalize(pixels, dim=1).cpu().numpy())
        return {k: np.concatenate(v).astype(np.float32) for k, v in collected.items()}
    begin = time.perf_counter()
    features = extract(train, union)
    test_features = extract(test, test_ids)
    positions = {int(i): j for j, i in enumerate(union)}
    val_pos = [positions[int(i)] for i in validation]
    np.savez_compressed(DATA / "evaluation.npz", val_x=features["resnet128"][val_pos],
        val_y=np.asarray(train.targets)[validation], test_x=test_features["resnet128"],
        test_y=np.asarray(test.targets)[test_ids])
    for seed, ids in enumerate(subsets):
        pos = [positions[int(i)] for i in ids]
        clean = np.asarray(train.targets)[ids]
        labels = clean.copy()
        rng = np.random.default_rng(10000 + seed)
        noisy = rng.choice(1000, 100, replace=False)
        labels[noisy] = (labels[noisy] + rng.integers(1, 10, 100)) % 10
        mask = np.zeros(1000, dtype=np.int8)
        mask[noisy] = 1
        np.savez_compressed(DATA / ("seed%d.npz" % seed), **{k: v[pos] for k, v in features.items()},
            labels=labels, noise=mask, source_indices=ids)
    dump(DATA / "manifest.json", dict(dataset="CIFAR10", frozen_encoder="ImageNet ResNet18",
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        data_seeds=[0, 1, 2], train_seeds=[0, 1, 2], graph_seeds=[0, 1, 2],
        training_indices=[ids.tolist() for ids in subsets], validation_train_indices=validation.tolist(),
        final_test_indices=test_ids.tolist(), feature_extraction_seconds=time.perf_counter() - begin,
        input_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in DATA.glob("*.npz")}))


def inputs(x, device):
    x = torch.from_numpy(x).to(device)
    return torch.cat([x, torch.ones(len(x), 1, device=device)], dim=1)


def head(seed, device):
    torch.manual_seed(seed)
    return torch.nn.Linear(513, 10, bias=False).to(device)


def sample_grad(net, x, y):
    residual = net(x).softmax(1) - (.9 * F.one_hot(y, 10) + .01)
    return (residual[:, :, None] * x[:, None, :]).flatten(1)


def trace(x, y, vx, vy, seed, directory):
    net = head(seed, x.device)
    opt = torch.optim.Adam(net.parameters(), lr=.001, weight_decay=.0001)
    with torch.no_grad():
        initial_grad = sample_grad(net, x, y).cpu().numpy()
    rng = np.random.default_rng(seed)
    all_ids, all_raw, ptr = [], [], [0]
    for epoch in range(5):
        order = rng.permutation(len(x))
        for start in range(0, len(x), 64):
            ids = order[start:start + 64]
            with torch.no_grad():
                grads = sample_grad(net, x[ids], y[ids])
                query = sample_grad(net, vx[:128], vy[:128]).mean(0)
                state = opt.state.get(net.weight, {})
                m = state.get("exp_avg", torch.zeros_like(net.weight)).flatten()
                v = state.get("exp_avg_sq", torch.zeros_like(net.weight)).flatten()
                t = int(state["step"].item()) + 1 if "step" in state else 1
                direction = ((.9 * m + .1 * grads) / (1 - .9 ** t)) / (
                    ((.999 * v + .001 * grads.square()) / (1 - .999 ** t)).sqrt() + 1e-8)
                raw = (.001 / len(ids) * (direction * query).sum(1)).cpu().numpy()
            all_ids.extend(ids)
            all_raw.extend(raw)
            ptr.append(len(all_ids))
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(net(x[ids]), y[ids], label_smoothing=.1).backward()
            opt.step()
    result = dict(ids=np.asarray(all_ids), raw=np.asarray(all_raw, dtype=np.float32),
                  ptr=np.asarray(ptr), initial_grad=initial_grad)
    np.savez_compressed(directory / "trajectory.npz", **result)
    return result


def graph(features, k, sym="directed", ann=0, seed=0):
    unit = features.astype(np.float32).copy()
    unit /= np.linalg.norm(unit, axis=1, keepdims=True).clip(1e-8)
    similarities = unit @ unit.T
    np.fill_diagonal(similarities, -np.inf)
    exact = np.argsort(-similarities, axis=1, kind="stable")[:, :k]
    if ann:
        insertion = np.random.default_rng(seed).permutation(len(unit))
        index = faiss.IndexHNSWFlat(unit.shape[1], 4)
        index.hnsw.efConstruction, index.hnsw.efSearch = 20, ann
        index.add(np.ascontiguousarray(unit[insertion]))
        _, found = index.search(unit, k + 1)
        neighbors = np.tile(np.arange(len(unit))[:, None], (1, k))
        for i, row in enumerate(found):
            row = insertion[row[row >= 0]]
            row = row[row != i]
            # A tiny search budget may return fewer than k valid neighbors.
            # Missing entries remain zero-weight self padding, not exact fills.
            neighbors[i, :min(k, len(row))] = row[:k]
    else:
        neighbors = exact
    recall = float(np.mean([len(set(a) & set(b)) / k for a, b in zip(exact, neighbors)]))
    weights = np.maximum(similarities[np.arange(len(unit))[:, None], neighbors], 0)
    weights /= weights.sum(1, keepdims=True).clip(1e-8)
    dense = np.zeros((len(unit), len(unit)), dtype=np.float32)
    dense[np.arange(len(unit))[:, None], neighbors] = weights
    if sym == "mean":
        dense = (dense + dense.T) * .5
    elif sym == "mutual":
        dense = (dense + dense.T) * .5 * ((dense > 0) & (dense.T > 0))
    sparse = csr_matrix(dense)
    width = max(1, int(np.diff(sparse.indptr).max()))
    nn = np.tile(np.arange(len(unit))[:, None], (1, width)).astype(np.int32)
    ww = np.zeros(nn.shape, dtype=np.float32)
    for i in range(len(unit)):
        a, b = sparse.indptr[i:i + 2]
        nn[i, :b - a], ww[i, :b - a] = sparse.indices[a:b], sparse.data[a:b]
    return nn, ww, recall, dense


def replay(trajectory, nn, ww, confidence, cfg):
    history = np.zeros(len(nn), dtype=np.float32)
    scores = np.zeros_like(history)
    for a, b in zip(trajectory["ptr"][:-1], trajectory["ptr"][1:]):
        ids, raw = trajectory["ids"][a:b], trajectory["raw"][a:b]
        neighbors, weights = nn[ids], ww[ids]
        diffusion = (weights * history[neighbors]).sum(1) * confidence[ids]
        local = np.zeros(len(nn), dtype=np.float32)
        local[ids] = raw
        pair = (weights * local[neighbors]).sum(1) * confidence[ids]
        def transform(term):
            if not cfg["scale"]:
                return term
            x, y = float(raw.std()), float(term.std())
            if x < 1e-12 or y < 1e-12:
                return np.zeros_like(term)
            return (term - term.mean()) * (x / (y + 1e-12))
        corrected = raw + cfg["alpha"] * transform(diffusion)
        if cfg["pair"]:
            corrected += .5 * cfg["alpha"] * transform(pair)
        scores[ids] += corrected
        if cfg["history"]:
            history[ids] += raw
    return scores


def fit_subset(x, y, tx, ty, selected, seed):
    net = head(7000 + seed, x.device)
    opt = torch.optim.Adam(net.parameters(), lr=.001, weight_decay=.0001)
    schedule = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=.1)
    rng = np.random.default_rng(7000 + seed)
    selected = np.sort(selected)
    for epoch in range(50):
        order = rng.permutation(selected)
        for start in range(0, len(order), 64):
            ids = order[start:start + 64]
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(net(x[ids]), y[ids], label_smoothing=.1).backward()
            opt.step()
        schedule.step()
    with torch.no_grad():
        logits = net(tx)
        losses = F.cross_entropy(logits, ty, reduction="none").cpu().numpy()
        predictions = logits.argmax(1).cpu().numpy()
    return dict(test_accuracy=float((predictions == ty.cpu().numpy()).mean()),
                test_loss=float(losses.mean())), predictions, losses


def worker(data_seed):
    device = torch.device("cuda:%d" % data_seed)
    torch.set_num_threads(2)
    faiss.omp_set_num_threads(1)
    source = np.load(DATA / ("seed%d.npz" % data_seed))
    evaluation = np.load(DATA / "evaluation.npz")
    x, vx, tx = [inputs(a, device) for a in (source["resnet128"], evaluation["val_x"], evaluation["test_x"])]
    y, vy, ty = [torch.from_numpy(a.astype(np.int64)).to(device) for a in (source["labels"], evaluation["val_y"], evaluation["test_y"])]
    noise = source["noise"]
    graph_cache = {}
    for train_seed in range(3):
        folder = OUT / ("data%d_train%d" % (data_seed, train_seed))
        folder.mkdir(exist_ok=True)
        if (folder / "complete.json").exists():
            continue
        trajectory = trace(x, y, vx, vy, train_seed, folder)
        rows, scores_saved, fits, gates = [], {}, {}, {}
        reference = graph(source["resnet128"], 10)[3]
        reference = (reference + reference.T) * .5
        full_metrics, predictions, losses = fit_subset(x, y, tx, ty, np.arange(1000), train_seed)
        dump(folder / "full_data_reference.json", full_metrics)
        np.savez_compressed(folder / "full_data_predictions.npz", predictions=predictions, losses=losses)
        for name, cfg in configs():
            key = (cfg["embedding"], cfg["k"], cfg["sym"], cfg["ann"], cfg["graph_seed"] if cfg["ann"] else 0)
            if key not in graph_cache:
                graph_cache[key] = graph(source[cfg["embedding"]], cfg["k"], cfg["sym"], cfg["ann"], cfg["graph_seed"])
            nn, ww, recall, dense = graph_cache[key]
            if key not in gates:
                gates[key] = confidence_gated_graph_weights(trajectory["initial_grad"], nn, ww, chunk_size=32)
            weights, confidence = gates[key] if cfg["gate"] else (ww, np.ones(1000, dtype=np.float32))
            scores = replay(trajectory, nn, weights, confidence, cfg)
            if name == "random":
                scores = np.random.default_rng(9000 + train_seed).normal(size=1000).astype(np.float32)
            selected, _ = diversity_select(scores, nn, ww, 200, lambda_fraction=cfg["lam"])
            if name == "random":
                selected = np.argsort(-scores, kind="stable")[:200]
            signature = hashlib.sha256(np.sort(selected).astype(np.int32).tobytes()).hexdigest()[:16]
            reused = signature in fits
            if not reused:
                metrics, predictions, losses = fit_subset(x, y, tx, ty, selected, train_seed)
                fits[signature] = metrics
                np.savez_compressed(folder / ("fit_" + signature + ".npz"),
                    selected=selected, predictions=predictions, losses=losses)
            mask = np.zeros(1000, dtype=np.float32)
            mask[selected] = 1
            row = dict(data_seed=data_seed, train_seed=train_seed, variant=name, **cfg,
                ann_recall=recall, noise_auroc=float(roc_auc_score(noise, -scores)),
                effective_degree=float((dense > 0).sum(1).mean()),
                selected_class_count=int(len(np.unique(source["labels"][selected]))),
                selected_noise_fraction=float(noise[selected].mean()),
                reference_coverage=float(((mask > 0) | (reference @ mask > 0)).mean()),
                reference_pair_affinity=float(mask @ reference @ mask / 2 / 200),
                unique_fit_id=signature, utility_reused=reused, **fits[signature])
            rows.append(row)
            scores_saved[name] = scores
            dump(folder / "results.json", rows)
            print(data_seed, train_seed, name, round(row["test_accuracy"], 4), flush=True)
        np.savez_compressed(folder / "scores.npz", **scores_saved)
        dump(folder / "complete.json", dict(cells=len(rows), actual_fits=len(fits), full_data_reference_fits=1))


def tests():
    torch.manual_seed(9)
    net = torch.nn.Linear(513, 10, bias=False)
    x = torch.randn(3, 513)
    y = torch.tensor([1, 5, 9])
    actual = sample_grad(net, x, y)
    for i in range(3):
        expected = torch.autograd.grad(F.cross_entropy(net(x[i:i + 1]), y[i:i + 1], label_smoothing=.1), net.weight)[0]
        torch.testing.assert_close(actual[i], expected.flatten(), atol=1e-6, rtol=1e-5)
    rng = np.random.default_rng(5)
    features = rng.normal(size=(40, 12)).astype(np.float32)
    nn, ww, recall, dense = graph(features, 5)
    assert recall == 1 and not np.any(nn == np.arange(40)[:, None])
    for sym in ("mean", "mutual"):
        d = graph(features, 5, sym)[3]
        np.testing.assert_array_equal(d, d.T)
    raw = rng.normal(size=40).astype(np.float32)
    trajectory = dict(ids=np.arange(40), raw=raw, ptr=np.arange(0, 41, 8))
    cfg = dict(configs()[0][1], alpha=0)
    np.testing.assert_array_equal(replay(trajectory, nn, ww, np.ones(40), cfg), raw)
    # With only first-order history disabled and pair disabled, raw scores survive.
    cfg.update(alpha=.2, history=False, pair=False)
    np.testing.assert_array_equal(replay(trajectory, nn, ww, np.ones(40), cfg), raw)
    first = graph(features, 5, ann=8, seed=2)[0]
    np.testing.assert_array_equal(first, graph(features, 5, ann=8, seed=2)[0])
    dump(OUT / "tests.json", dict(status="passed", checks=["analytic_gradient_vs_autograd", "self_exclusion",
        "symmetry", "zero_alpha", "no_history_no_pair", "seeded_ANN_reproducibility"]))


def summarize():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows, totals = [], []
    for ds in range(3):
        for ts in range(3):
            folder = OUT / ("data%d_train%d" % (ds, ts))
            values = json.loads((folder / "results.json").read_text())
            assert len(values) == len(configs())
            rows.extend(values)
            totals.append(json.loads((folder / "complete.json").read_text()))
            with np.load(folder / "scores.npz") as scores:
                for seed in (1, 2):
                    np.testing.assert_array_equal(scores["full"], scores["exact_seed%d" % seed])
                for key in scores.files:
                    assert scores[key].shape == (1000,) and np.isfinite(scores[key]).all()
    csv_write(OUT / "all_results.csv", rows)
    metrics = ("test_accuracy", "test_loss", "noise_auroc", "selected_noise_fraction", "reference_coverage", "reference_pair_affinity", "ann_recall")
    summary = []
    for name, cfg in configs():
        group = [r for r in rows if r["variant"] == name]
        item = dict(variant=name, **cfg)
        for metric in metrics:
            item[metric + "_mean"] = float(np.mean([r[metric] for r in group]))
            item[metric + "_std"] = float(np.std([r[metric] for r in group], ddof=1))
        summary.append(item)
    csv_write(OUT / "summary.csv", summary)
    stability = []
    for ds in range(3):
        data = [np.load(OUT / ("data%d_train%d/scores.npz" % (ds, ts))) for ts in range(3)]
        for name, cfg in configs():
            correlations = [float(spearmanr(data[a][name], data[b][name]).correlation) for a, b in ((0, 1), (0, 2), (1, 2))]
            stability.append(dict(data_seed=ds, variant=name, mean_pairwise_spearman=float(np.mean(correlations))))
        for d in data:
            d.close()
    csv_write(OUT / "training_seed_stability.csv", stability)
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "svg.fonttype": "none"})
    def save(fig, name):
        fig.tight_layout()
        for suffix in ("png", "pdf", "svg"):
            fig.savefig(OUT / (name + "." + suffix), dpi=180)
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
    grids = [r for r in summary if r["variant"] == "full" or r["variant"].startswith("k")]
    lo, hi = min(r["test_accuracy_mean"] for r in grids), max(r["test_accuracy_mean"] for r in grids)
    for ax, k in zip(axes, (5, 10, 20)):
        grid = np.array([[next(r["test_accuracy_mean"] for r in grids if r["k"] == k and r["alpha"] == a and r["lam"] == lam) for lam in (0, .05, .1)] for a in (0, .2, .4)])
        ax.imshow(grid, vmin=lo, vmax=hi, cmap="viridis")
        for (i, j), value in np.ndenumerate(grid):
            ax.text(j, i, "%.3f" % value, ha="center", va="center", color="white" if value < (lo + hi) / 2 else "black")
        ax.set(xticks=range(3), xticklabels=[0, .05, .1], yticks=range(3), yticklabels=[0, .2, .4], xlabel="lambda fraction", ylabel="alpha", title="k=%d; test accuracy" % k)
    save(fig, "parameter_heatmaps")
    names = ["full", "no_gate", "no_history", "no_pair", "no_scale", "k10_a0.2_l0", "resnet32", "pixels8", "sym_mean", "sym_mutual"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, metric in zip(axes, ("test_accuracy", "noise_auroc")):
        group = [next(r for r in summary if r["variant"] == name) for name in names]
        ax.errorbar(range(len(names)), [r[metric + "_mean"] for r in group],
            yerr=[r[metric + "_std"] for r in group], fmt="o", capsize=3, color="#247a60")
        ax.set(xticks=range(len(names)), xticklabels=names, ylabel=metric)
        ax.tick_params(axis="x", labelrotation=55)
    save(fig, "module_ablations")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ("test_accuracy", "noise_auroc")):
        group = [next(r for r in summary if r["variant"] == "ann%d" % ef) for ef in (2, 8, 32, 128)]
        ax.plot([r["ann_recall_mean"] for r in group], [r[metric + "_mean"] for r in group], "o-", color="#b64b56")
        for r in group:
            ax.annotate("ef=%d" % r["ann"], (r["ann_recall_mean"], r[metric + "_mean"]), xytext=(3, 5), textcoords="offset points", fontsize=8)
        ax.set(xlabel="Measured ANN Recall@10", ylabel=metric)
    save(fig, "ann_recall_quality")
    seed_rows = []
    for ds in range(3):
        for ts in range(3):
            for gs in range(3):
                name = "ann8" if gs == 0 else "ann8_seed%d" % gs
                row = next(r for r in rows if r["data_seed"] == ds and r["train_seed"] == ts and r["variant"] == name)
                seed_rows.append(row)
    csv_write(OUT / "seed_factorial.csv", seed_rows)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, factor in zip(axes, ("data_seed", "train_seed", "graph_seed")):
        for level in range(3):
            vals = [r["test_accuracy"] for r in seed_rows if r[factor] == level]
            ax.scatter(np.full(len(vals), level) + np.linspace(-.08, .08, len(vals)), vals, s=20)
            ax.plot([level - .2, level + .2], [np.mean(vals)] * 2, color="black")
        ax.set(xticks=[0, 1, 2], xlabel=factor, ylabel="Test accuracy")
    save(fig, "separated_seeds")
    lines = ["# Appendix D: CIFAR Sensitivity", "",
        "Frozen ImageNet ResNet18 features, trainable linear head only. 3 data seeds crossed with 3 training seeds. 43 fixed graph configurations plus a random reference per trajectory. Each selection retains 200/1000 examples and is genuinely retrained for 50 head-training epochs. Identical selected sets within a seed pair reuse one fit and are explicitly marked; they are not independent training runs.", "",
        "Full denotes the optional initial-gradient-gated extension; no_gate is the semantic-only paper weighting. Gate gradients are computed at initialization, not taken from future trajectory states. Scores replay the exact recorded five-epoch first-order Adam trajectory; no HVP or second-order cache is used.", "",
        "Validation is 1000 held-out CIFAR TRAIN images (128 used by attribution); test is 2000 official TEST images. There is no validation/test overlap. No hyperparameter was selected on the final test outcomes; this is a descriptive sweep, not a tuned main result.", "",
        "Mean (sample SD) across the 9 crossed data/training cells; these are NOT 9 independent dataset samples. All outcomes, including null and adverse ones, are retained.", "",
        "| Variant | Test accuracy | Test loss | Noise AUROC | Selected noise | Coverage | ANN recall |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary:
        values = ["%.5f (%.5f)" % (row[m + "_mean"], row[m + "_std"]) for m in ("test_accuracy", "test_loss", "noise_auroc", "selected_noise_fraction", "reference_coverage", "ann_recall")]
        lines.append("| %s | %s |" % (row["variant"], " | ".join(values)))
    lines += ["", "## Interpretation boundaries", "",
        "- This is not Table 5 full-network training. Alternative graph embeddings change only the graph; the classification representation remains resnet128.",
        "- Directed/mean/mutual refer to propagation adjacency. Selection always uses the symmetric redundancy objective. Diversity is evaluated on the SAME reference graph for every configuration.",
        "- HNSW uses M=4, efConstruction=20, one thread. Graph seed controls insertion permutation; training/data seeds stay fixed. Exact graph seeds are a deterministic negative control, checked for bitwise-identical scores.",
        "- At very low efSearch, HNSW may return fewer than k valid neighbors. Missing neighbors carry zero weight and are NOT filled from the exact graph. Effective degree is recorded alongside recall.",
        "- Seed-factor results comprise a fully crossed 3 x 3 x 3 ANN design. Marginal comparisons are descriptive; data-seed subsets can overlap and pairwise rank correlations are dependent.",
        "- Existing Pile results remain available under efficiency_scalability_ablations and are not replaced by this modality change. The Pile 1k online study had near-chance noise AUROC and all selected-subset losses worse than its pretrained reference.",
        "- Retention is fixed at 20%. Larger/full-network or other-budget conclusions require further experiments. No artificial variance is added to deterministic controls."]
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    dump(OUT / "validation.json", dict(status="passed", trajectories=9, configurations=len(configs()),
        result_cells=len(rows), unique_retraining_fits=sum(t["actual_fits"] for t in totals),
        graph_seed_factorial_cells=len(seed_rows), exact_seed_control="bitwise equal"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("all", "prepare", "worker", "summarize", "test"))
    parser.add_argument("--data-seed", type=int, default=0)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    faiss.omp_set_num_threads(1)
    if args.mode == "worker":
        worker(args.data_seed)
    elif args.mode == "prepare":
        prepare()
    elif args.mode == "summarize":
        summarize()
    elif args.mode == "test":
        tests()
    else:
        tests()
        if not (DATA / "manifest.json").exists():
            prepare()
        dump(OUT / "protocol.json", dict(configurations=configs(), budget=.2, score_epochs=5,
            retrain_epochs=50, model="frozen ImageNet ResNet18 + normalized-feature linear head",
            data_seeds=[0, 1, 2], train_seeds=[0, 1, 2], graph_seeds=[0, 1, 2],
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            torch=torch.__version__, numpy=np.__version__, faiss=faiss.__version__))
        processes = []
        for seed in range(3):
            log = (OUT / ("data%d.log" % seed)).open("a")
            proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "worker", "--data-seed", str(seed)], stdout=log, stderr=subprocess.STDOUT)
            processes.append((proc, log))
        codes = []
        for proc, log in processes:
            codes.append(proc.wait())
            log.close()
        if any(codes):
            raise RuntimeError("Worker failure: %s; inspect logs" % codes)
        summarize()


if __name__ == "__main__":
    main()
