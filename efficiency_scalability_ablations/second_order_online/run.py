"""Matched-trajectory HVP versus genuinely Hessian-free online graph benchmark."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "pile_main_100k"))
import numpy as np
import psutil
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from common import (corrupt_rows, diversity_select, example_losses, model,
                    replay_graph_trajectory, seed_all, tokenizer, trainable)
from graph_utils import gpu_chunked_knn

OUT = HERE / "results"
DATA = HERE / "data"
METHODS = ("inrun2", "adam2", "graph", "adam1")
LR = 6e-4


def dump(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def csv_write(path, rows):
    with path.open("w", newline="") as handle:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class OnlineGraph:
    def __init__(self, neighbors, weights, alpha=.2):
        self.neighbors, self.weights, self.alpha = neighbors, weights, alpha
        self.history = np.zeros(len(neighbors), dtype=np.float32)
        self.pair_edges = 0

    def update(self, ids, raw):
        raw = np.asarray(raw, dtype=np.float32)
        nn, ww = self.neighbors[ids], self.weights[ids]
        diffusion = (ww * self.history[nn]).sum(1)
        positions = {int(index): pos for pos, index in enumerate(ids)}
        pair = np.zeros(len(ids), dtype=np.float32)
        for row, (neighbors, weights) in enumerate(zip(nn, ww)):
            for neighbor, weight in zip(neighbors, weights):
                pos = positions.get(int(neighbor))
                if pos is not None:
                    pair[row] += float(weight) * float(raw[pos])
                    self.pair_edges += 1

        def match(term):
            a, b = float(raw.std()), float(term.std())
            if a < 1e-12 or b < 1e-12:
                return np.zeros_like(term)
            return (term - float(term.mean())) * (a / (b + 1e-12))

        corrected = raw + self.alpha * match(diffusion)
        corrected += .5 * self.alpha * match(pair)
        self.history[ids] += raw
        return corrected


@contextmanager
def grad_audit(method):
    """Fail at runtime if Graph requests higher-order autograd graphs or an HVP."""
    original = torch.autograd.grad
    counts = dict(grad_calls=0, create_graph_calls=0, hvp_calls=0)

    def wrapped(*args, **kwargs):
        counts["grad_calls"] += 1
        create = kwargs.get("create_graph", args[4] if len(args) > 4 else False)
        if create:
            counts["create_graph_calls"] += 1
            if method in ("graph", "adam1"):
                raise AssertionError("First-order path requested create_graph=True")
        return original(*args, **kwargs)

    torch.autograd.grad = wrapped
    try:
        yield counts
    finally:
        torch.autograd.grad = original


class RSSMonitor:
    def __enter__(self):
        self.proc = psutil.Process()
        self.peak = self.proc.memory_info().rss
        self.stop = threading.Event()
        def sample():
            while not self.stop.wait(.005):
                self.peak = max(self.peak, self.proc.memory_info().rss)
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.peak = max(self.peak, self.proc.memory_info().rss)
        self.stop.set()
        self.thread.join()


def prepare():
    DATA.mkdir(parents=True, exist_ok=True)
    train_indices = np.random.default_rng(2027).permutation(100000)[:1000]
    val_indices = np.random.default_rng(2028).permutation(10000)[:144]
    tok = tokenizer()
    def read_selected(split, indices):
        wanted = set(map(int, indices))
        found = {}
        with (ROOT / "data/Pile_large" / (split + ".jsonl")).open() as handle:
            for i, line in enumerate(handle):
                if i in wanted:
                    row = json.loads(line)
                    found[i] = dict(text=row["text"], domain=row.get("meta", {}).get("pile_set_name", "Unknown"))
                if len(found) == len(wanted):
                    break
        return [found[int(i)] for i in indices]
    train = read_selected("train", train_indices)
    validation = read_selected("val", val_indices)
    def tokens(rows):
        batch = tok([r["text"] for r in rows], padding="max_length", truncation=True,
                    max_length=128, return_tensors="np")
        return dict(ids=batch["input_ids"], mask=batch["attention_mask"])
    for seed in range(3):
        for scenario in ("clean", "noise"):
            rows, labels = (corrupt_rows(train, .1, seed + 991) if scenario == "noise"
                            else (train, np.zeros(len(train), dtype=np.int8)))
            np.savez_compressed(DATA / ("%s_seed%d.npz" % (scenario, seed)),
                                **tokens(rows), labels=labels,
                                domains=np.asarray([r["domain"] for r in rows]))
    np.savez_compressed(DATA / "validation.npz", **tokens(validation))
    dump(DATA / "manifest.json", dict(train_indices=train_indices.tolist(),
        utility_val_indices=val_indices[:16].tolist(), evaluation_val_indices=val_indices[16:].tolist(),
        train_corpora=1000, sequence_length=128, corruption="common.corrupt_rows, fraction=.1, seed+991",
        train_source="data/Pile_large/train.jsonl", val_source="data/Pile_large/val.jsonl",
        tokens={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in DATA.glob("*.npz")}))


def load_arrays(path, device):
    with np.load(path) as cache:
        return {"input_ids": torch.from_numpy(cache["ids"].astype(np.int64)).to(device),
                "attention_mask": torch.from_numpy(cache["mask"].astype(np.int64)).to(device)}


def take(batch, ids):
    return {key: value[ids] for key, value in batch.items()}


def optimizer_for(net):
    return torch.optim.AdamW(trainable(net), lr=LR, betas=(.9, .95), weight_decay=.1)


def timed(device, fn):
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    result = fn()
    torch.cuda.synchronize(device)
    return result, time.perf_counter() - start


def step(net, optimizer, batch, validation, ids, method, graph, device, audit):
    params = trainable(net)
    timings = {}
    def phase(name, fn):
        result, elapsed = timed(device, fn)
        timings[name] = elapsed
        return result
    net.train()
    def query_gradient():
        loss = example_losses(net, validation).mean()
        parts = torch.autograd.grad(loss, params, create_graph=method in ("inrun2", "adam2"))
        return torch.cat([part.reshape(-1) for part in parts])
    query = phase("query_s", query_gradient)
    def sample_gradients():
        losses = example_losses(net, batch)
        values = []
        for pos, loss in enumerate(losses):
            parts = torch.autograd.grad(loss, params, retain_graph=pos + 1 < len(losses))
            values.append(torch.cat([p.reshape(-1) for p in parts]).detach())
        return torch.stack(values)
    grads = phase("sample_s", sample_gradients)
    def directions():
        if method == "inrun2":
            return grads
        moments, variances, t = [], [], 1
        for parameter in params:
            state = optimizer.state.get(parameter, {})
            moments.append(state.get("exp_avg", torch.zeros_like(parameter)).reshape(-1))
            variances.append(state.get("exp_avg_sq", torch.zeros_like(parameter)).reshape(-1))
            if "step" in state:
                t = int(state["step"].item()) + 1
        m, v = torch.cat(moments), torch.cat(variances)
        next_m = .9 * m[None] + .1 * grads
        next_v = .95 * v[None] + .05 * grads.square()
        return (next_m / (1 - .9 ** t)) / ((next_v / (1 - .95 ** t)).sqrt() + 1e-8)
    direction = phase("direction_s", directions)
    scale = LR / len(ids)
    raw = phase("first_s", lambda: scale * (direction * query.detach()).sum(1))
    def interaction():
        if method in ("inrun2", "adam2"):
            audit["hvp_calls"] += 1
            total_step = -scale * direction.sum(0)
            parts = torch.autograd.grad((query * total_step).sum(), params)
            hvp = torch.cat([p.reshape(-1) for p in parts]).detach()
            curvature = -.5 * ((-scale * direction) * hvp).sum(1)
            a, b = raw.std(), curvature.std()
            if b > .25 * a and b > 1e-12:
                curvature.mul_(.25 * a / b)
            return raw + curvature
        if method == "graph":
            assert not query.requires_grad and not direction.requires_grad
            return graph.update(ids, raw.detach().cpu().numpy())
        return raw
    values = phase("interaction_s", interaction)
    def export():
        final = values if isinstance(values, np.ndarray) else values.detach().cpu().numpy()
        return final.copy(), raw.detach().cpu().numpy().copy()
    final, raw_cpu = phase("export_s", export)
    # Release attribution tensors before the identical optimizer update.
    del query, grads, direction, raw, values
    def update():
        optimizer.zero_grad(set_to_none=True)
        loss = example_losses(net, batch).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        return float(loss.detach())
    train_loss = phase("update_s", update)
    return final, raw_cpu, timings, train_loss


def run_cell(seed, scenario, method, gpu):
    device = torch.device("cuda:%d" % gpu)
    torch.set_num_threads(2)
    seed_all(seed)
    net = model(device)
    initial = {k: v.detach().clone() for k, v in net.transformer.ln_f.state_dict().items()}
    batch = load_arrays(DATA / ("%s_seed%d.npz" % (scenario, seed)), device)
    validation = take(load_arrays(DATA / "validation.npz", device), slice(0, 16))
    order = np.random.default_rng(seed).permutation(1000)
    graph, nn, ww = None, None, None
    prep = dict(embedding_s=0., knn_s=0., peak_cuda_mib=0., rss_peak_mib=0.)
    if method == "graph":
        torch.cuda.reset_peak_memory_stats(device)
        def semantic_features():
            net.eval()
            arrays = []
            with torch.no_grad():
                for start in range(0, 1000, 32):
                    local = take(batch, slice(start, start + 32))
                    hidden = net.transformer(**local).last_hidden_state.float()
                    mask = local["attention_mask"].unsqueeze(-1)
                    vectors = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
                    arrays.append(torch.nn.functional.normalize(vectors, dim=1).cpu().numpy().astype(np.float16))
            net.train()
            return np.concatenate(arrays)
        with RSSMonitor() as mem:
            features, prep["embedding_s"] = timed(device, semantic_features)
            (nn, ww), prep["knn_s"] = timed(device, lambda: gpu_chunked_knn(features, device, k=10))
        prep["peak_cuda_mib"] = torch.cuda.max_memory_allocated(device) / 1024 ** 2
        prep["rss_peak_mib"] = mem.peak / 1024 ** 2
        graph = OnlineGraph(nn, ww)
    optimizer = optimizer_for(net)
    with grad_audit(method) as warm_audit:
        for start in (0, 8):
            ids = order[start:start + 8]
            step(net, optimizer, take(batch, ids), validation, ids, method, graph, device, warm_audit)
    net.transformer.ln_f.load_state_dict(initial)
    optimizer = optimizer_for(net)
    net.zero_grad(set_to_none=True)
    graph = OnlineGraph(nn, ww) if method == "graph" else None
    seed_all(seed)
    scores, first = np.zeros(1000, dtype=np.float32), np.zeros(1000, dtype=np.float32)
    rows, losses = [], []
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    baseline_memory = torch.cuda.memory_allocated(device)
    with RSSMonitor() as mem, grad_audit(method) as audit:
        start_time = time.perf_counter()
        for number, start in enumerate(range(0, 1000, 8)):
            ids = order[start:start + 8]
            scores[ids], first[ids], stages, loss = step(
                net, optimizer, take(batch, ids), validation, ids, method, graph, device, audit)
            rows.append(dict(step=number, **stages))
            losses.append(loss)
            if (number + 1) % 25 == 0:
                print("seed", seed, scenario, method, "steps", number + 1, flush=True)
        torch.cuda.synchronize(device)
        online_time = time.perf_counter() - start_time
    assert audit["hvp_calls"] == (125 if method in ("inrun2", "adam2") else 0)
    assert audit["create_graph_calls"] == audit["hvp_calls"]
    totals = {key: sum(row[key] for row in rows) for key in rows[0] if key != "step"}
    attribution = sum(value for key, value in totals.items() if key != "update_s")
    final_parameters = torch.cat([p.detach().reshape(-1) for p in trainable(net)]).cpu().numpy()
    output = OUT / "cells" / ("%s_seed%d_%s" % (scenario, seed, method))
    np.savez_compressed(str(output) + ".npz", scores=scores, first=first,
        final_parameters=final_parameters, training_losses=np.asarray(losses), order=order,
        **({"neighbors": nn, "weights": ww} if method == "graph" else {}))
    with np.load(DATA / ("%s_seed%d.npz" % (scenario, seed))) as data:
        labels = data["labels"]
    dump(Path(str(output) + ".json"), dict(seed=seed, scenario=scenario, method=method,
        n=1000, steps=125, batch_size=8, gpu=gpu, hardware=torch.cuda.get_device_name(device),
        trainable_parameters=len(final_parameters), online_wall_s=online_time,
        attribution_components_s=attribution,
        attribution_with_graph_prep_s=attribution + prep["embedding_s"] + prep["knn_s"],
        pipeline_with_graph_prep_s=online_time + prep["embedding_s"] + prep["knn_s"],
        peak_cuda_mib=torch.cuda.max_memory_allocated(device) / 1024 ** 2,
        extra_cuda_mib=(torch.cuda.max_memory_allocated(device) - baseline_memory) / 1024 ** 2,
        rss_peak_mib=mem.peak / 1024 ** 2, preparation=prep, audit=audit, stages=totals,
        pair_edges=graph.pair_edges if graph is not None else None, step_timings=rows,
        noise_auroc=float(roc_auc_score(labels, -scores)) if scenario == "noise" else None))
    print("DONE", seed, scenario, method, round(online_time, 2), flush=True)


def quality(seed, gpu):
    device = torch.device("cuda:%d" % gpu)
    torch.set_num_threads(2)
    train = load_arrays(DATA / ("clean_seed%d.npz" % seed), device)
    evaluation = take(load_arrays(DATA / "validation.npz", device), slice(16, 144))
    graph_cache = np.load(OUT / "cells" / ("clean_seed%d_graph.npz" % seed))
    nn, ww = graph_cache["neighbors"], graph_cache["weights"]
    results = []
    def evaluate(net):
        net.eval()
        values = []
        with torch.no_grad():
            for start in range(0, 128, 8):
                values.extend(example_losses(net, take(evaluation, slice(start, start + 8))).cpu().tolist())
        return float(np.mean(values))
    for method in METHODS + ("graph_diverse",):
        source = "graph" if method == "graph_diverse" else method
        with np.load(OUT / "cells" / ("clean_seed%d_%s.npz" % (seed, source))) as cache:
            scores = cache["scores"]
        chosen, selection_time = timed(device, lambda: diversity_select(
            scores, nn, ww, 100, lambda_fraction=.05 if method == "graph_diverse" else 0))
        indices, diag = chosen
        seed_all(seed + 7000)
        net = model(device)
        initial_loss = evaluate(net)
        optimizer = optimizer_for(net)
        net.train()
        indices = np.random.default_rng(seed + 7000).permutation(np.sort(indices))
        start_time = time.perf_counter()
        for start in range(0, 100, 8):
            optimizer.zero_grad(set_to_none=True)
            loss = example_losses(net, take(train, indices[start:start + 8])).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable(net), 1.)
            optimizer.step()
        torch.cuda.synchronize(device)
        training_time = time.perf_counter() - start_time
        val_loss = evaluate(net)
        results.append(dict(seed=seed, method=method, budget=.1, selected_count=100,
            validation_loss=val_loss, initial_validation_loss=initial_loss,
            validation_loss_improvement=initial_loss - val_loss,
            selection_s=selection_time, utility_training_s=training_time,
            selected_indices=indices.tolist(), **diag))
        print("QUALITY", seed, method, val_loss, flush=True)
        del net, optimizer
    graph_cache.close()
    dump(OUT / "quality" / ("seed%d.json" % seed), results)


def tests():
    rng = np.random.default_rng(27)
    n, k = 40, 5
    nn = np.asarray([rng.choice([j for j in range(n) if j != i], k, replace=False) for i in range(n)])
    ww = rng.uniform(size=(n, k)).astype(np.float32)
    ww /= ww.sum(1, keepdims=True)
    raw = rng.normal(size=n).astype(np.float32)
    for seed in (0, 1, 2):
        graph = OnlineGraph(nn, ww)
        order = np.random.default_rng(seed).permutation(n)
        actual = np.zeros(n, dtype=np.float32)
        for start in range(0, n, 8):
            ids = order[start:start + 8]
            assert np.all(graph.history[order[start:]] == 0)
            actual[ids] = graph.update(ids, raw[ids])
        expected = replay_graph_trajectory(raw, nn, ww, np.ones(n, dtype=np.float32), seed, include_pair=True)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(graph.history, raw)
    np.testing.assert_array_equal(OnlineGraph(nn, ww, alpha=0).update(np.arange(8), raw[:8]), raw[:8])
    x = torch.tensor([.4, -.7], dtype=torch.float64, requires_grad=True)
    loss = (x ** 4).sum()
    grad = torch.autograd.grad(loss, x, create_graph=True)[0]
    v = torch.tensor([.2, .3], dtype=torch.float64)
    hvp = torch.autograd.grad((grad * v).sum(), x)[0]
    torch.testing.assert_close(hvp, 12 * x.detach().square() * v)
    blocked = False
    try:
        with grad_audit("graph"):
            torch.autograd.grad((x ** 2).sum(), x, create_graph=True)
    except AssertionError:
        blocked = True
    assert blocked
    dump(OUT / "tests.json", dict(status="passed", checks=["online_vs_replay_three_seeds",
        "no_future_history", "zero_alpha", "analytic_HVP", "Graph_higher_order_guard"]))


def summarize():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cells = [json.loads(p.read_text()) for p in sorted((OUT / "cells").glob("*.json"))]
    assert len(cells) == 24
    assert {(r["seed"], r["scenario"], r["method"]) for r in cells} == {
        (s, scenario, method) for s in range(3) for scenario in ("clean", "noise") for method in METHODS}
    parity = []
    quality_rows = [r for p in sorted((OUT / "quality").glob("*.json")) for r in json.loads(p.read_text())]
    assert len(quality_rows) == 15
    for seed in range(3):
        for scenario in ("clean", "noise"):
            data = {method: np.load(OUT / "cells" / ("%s_seed%d_%s.npz" % (scenario, seed, method))) for method in METHODS}
            for method in METHODS:
                np.testing.assert_array_equal(data[method]["order"], data["adam1"]["order"])
                np.testing.assert_array_equal(data[method]["final_parameters"], data["adam1"]["final_parameters"])
                np.testing.assert_array_equal(data[method]["training_losses"], data["adam1"]["training_losses"])
            np.testing.assert_array_equal(data["graph"]["first"], data["adam1"]["scores"])
            np.testing.assert_array_equal(data["adam2"]["first"], data["adam1"]["scores"])
            parity.append(dict(seed=seed, scenario=scenario, status="bitwise_equal",
                graph_adam2_spearman=float(spearmanr(data["graph"]["scores"], data["adam2"]["scores"]).correlation),
                graph_adam1_spearman=float(spearmanr(data["graph"]["scores"], data["adam1"]["scores"]).correlation)))
            for item in data.values():
                item.close()
    dump(OUT / "trajectory_checks.json", parity)
    csv_write(OUT / "quality_all.csv", quality_rows)
    flat = [dict(**{k: v for k, v in r.items() if k not in ("stages", "step_timings", "preparation", "audit")},
                 **r["stages"], **r["audit"], embedding_s=r["preparation"]["embedding_s"], knn_s=r["preparation"]["knn_s"])
            for r in cells]
    csv_write(OUT / "timing_all.csv", flat)
    aggregates = []
    metrics = ("interaction_s", "query_s", "attribution_components_s", "attribution_with_graph_prep_s",
               "online_wall_s", "pipeline_with_graph_prep_s", "peak_cuda_mib", "extra_cuda_mib", "rss_peak_mib", "embedding_s", "knn_s")
    for scenario in ("clean", "noise"):
        for method in METHODS:
            group = [r for r in flat if r["scenario"] == scenario and r["method"] == method]
            aggregate = dict(scenario=scenario, method=method)
            for metric in metrics + (("noise_auroc",) if scenario == "noise" else ()):
                values = [r[metric] for r in group]
                aggregate[metric + "_mean"] = float(np.mean(values))
                aggregate[metric + "_std"] = float(np.std(values, ddof=1))
            aggregates.append(aggregate)
    csv_write(OUT / "timing_summary.csv", aggregates)
    speedups = []
    for scenario in ("clean", "noise"):
        for seed in range(3):
            by_method = {r["method"]: r for r in flat if r["scenario"] == scenario and r["seed"] == seed}
            for baseline in ("inrun2", "adam2"):
                row = dict(scenario=scenario, seed=seed, baseline=baseline)
                for metric in ("interaction_s", "attribution_with_graph_prep_s", "pipeline_with_graph_prep_s"):
                    row[metric + "_speedup"] = by_method[baseline][metric] / by_method["graph"][metric]
                speedups.append(row)
    csv_write(OUT / "paired_speedups.csv", speedups)
    quality_summary = []
    for method in METHODS + ("graph_diverse",):
        group = [r for r in quality_rows if r["method"] == method]
        row = dict(method=method)
        for metric in ("validation_loss", "validation_loss_improvement", "graph_coverage", "selected_pair_affinity_per_item"):
            row[metric + "_mean"] = float(np.mean([r[metric] for r in group]))
            row[metric + "_std"] = float(np.std([r[metric] for r in group], ddof=1))
        quality_summary.append(row)
    csv_write(OUT / "quality_summary.csv", quality_summary)
    def fmt(row, key, digits=3):
        return "%.*f (%.*f)" % (digits, row[key + "_mean"], digits, row[key + "_std"])
    report = ["# True HVP vs Hessian-Free Graph", "",
        "1000 Pile corpora, GPT-2 final LayerNorm only, 128 tokens, 125 AdamW updates (batch=8), three seeds. All four methods follow bitwise-identical optimizer trajectories within each seed/scenario. Graph consumes first-order Adam-aware scores online and makes ZERO higher-order gradient/HVP calls. No cached second-order scores or future gradients are read.", "",
        "Each HVP baseline computes ONE HVP per batch, not a dense Hessian or one HVP per example. SGD-style In-Run attribution is evaluated along the same AdamW training trajectory. The existing 25% curvature-SD cap is retained for both HVP baselines. Graph uses the paper's semantic weights without the optional gradient gate.", "",
        "## Clean Timing", "",
        "Mean (sample SD), seconds. Interaction includes HVP plus curvature correction, or graph diffusion + pair correction including device-to-host score transfer. It excludes the baseline's additional differentiable-query preparation, which IS included in attribution and pipeline totals.", "",
        "| Method | Interaction | Attribution + graph prep | Training/attribution pipeline + graph prep | Online CUDA peak MiB |",
        "|---|---:|---:|---:|---:|"]
    for row in aggregates:
        if row["scenario"] == "clean":
            report.append("| %s | %s | %s | %s | %s |" % (row["method"], fmt(row, "interaction_s"),
                fmt(row, "attribution_with_graph_prep_s"), fmt(row, "pipeline_with_graph_prep_s"), fmt(row, "peak_cuda_mib")))
    report += ["", "Graph prep includes a fresh full-corpus embedding forward pass and exact KNN. Attribution totals exclude the common optimizer-update pass; pipeline totals include it. Model loading, tokenization, two warmup steps and utility retraining are excluded for all arms. Graph preparation peak memory and CPU RSS are retained in raw JSON. CPU RSS is sampled every 5 ms.", "",
        "## Quality", "",
        "Noise AUROC: separate 10%-corrupted training run, three seeds. Subset validation loss: retrain from the SAME pretrained initialization on 100 selected clean examples, one pass, 13 updates; evaluate on 128 disjoint validation examples (not the 16 attribution utility examples). Top-k is used for ALL primary methods to isolate score quality. Graph-diverse is a separate lambda=.05 selection arm. Loss is mean per-example token-averaged NLL.", "",
        "| Method | Noise AUROC | Subset validation loss | Reference coverage |",
        "|---|---:|---:|---:|"]
    for row in quality_summary:
        source = "graph" if row["method"] == "graph_diverse" else row["method"]
        noise = next(r for r in aggregates if r["scenario"] == "noise" and r["method"] == source)
        report.append("| %s | %s | %s | %s |" % (row["method"], fmt(noise, "noise_auroc", 5),
            fmt(row, "validation_loss", 5), fmt(row, "graph_coverage", 5)))
    report += ["", "## Boundaries", "",
        "- This is a real 1k online/training benchmark, not a measured full 100k trajectory. Previous 1k/10k/50k/100k cached-graph scalability results remain separate; do not extrapolate measured end-to-end speedups to 100k.",
        "- Only 1536 LayerNorm parameters are differentiated. Full-parameter GPT-2/CIFAR conclusions need additional experiments. Higher-order autograd is real but restricted to this parameter set.",
        "- The graph interaction is a surrogate, not a numerically equivalent HVP. Score agreement, AUROC and held-out subset loss must be reported with speed; no outcome-based tuning or filtering is performed.",
        "- Stage synchronization adds instrumentation overhead. Three seeds use separate RTX 3090 devices, with method order rotated within seed; methods share the same device within each paired comparison.",
        "- A large interaction-only speedup need not imply a large end-to-end speedup because common gradient extraction/training can dominate. Report both, including one-time graph prep.",
        "- The new Hessian-free Graph is distinct from the old Graph-v2 path that consumed Adam-aware second-order caches. Old performance numbers cannot be relabeled as Hessian-free results."]
    (OUT / "report.md").write_text("\n".join(report) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    clean = [next(r for r in aggregates if r["scenario"] == "clean" and r["method"] == m) for m in METHODS]
    labels = ["In-Run 2", "Adam 2", "Graph", "Adam 1"]
    colors = ["#667c91", "#b95361", "#288065", "#b4983b"]
    for ax, metric, title in zip(axes, ("interaction_s", "attribution_with_graph_prep_s", "pipeline_with_graph_prep_s"),
        ("Interaction (log scale)", "Attribution + graph prep", "Training + attribution + graph prep")):
        ax.bar(labels, [r[metric + "_mean"] for r in clean], color=colors,
               yerr=[r[metric + "_std"] for r in clean], capsize=3)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Seconds / 1000 corpora")
        ax.tick_params(axis="x", labelrotation=20)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_yscale("log")
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUT / ("second_order_efficiency." + suffix), dpi=180)
    plt.close(fig)
    dump(OUT / "validation.json", dict(status="passed", timing_cells=24, utility_cells=15,
        matched_trajectories=6, graph_hvp_calls=0, graph_create_graph_calls=0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "test", "cell", "worker", "quality", "summarize", "all"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--scenario", default="clean", choices=("clean", "noise"))
    parser.add_argument("--method", default="graph", choices=METHODS)
    args = parser.parse_args()
    for folder in (OUT, OUT / "cells", OUT / "quality"):
        folder.mkdir(parents=True, exist_ok=True)
    if args.mode == "prepare":
        prepare()
    elif args.mode == "test":
        tests()
    elif args.mode == "cell":
        run_cell(args.seed, args.scenario, args.method, args.gpu)
    elif args.mode == "quality":
        quality(args.seed, args.gpu)
    elif args.mode == "summarize":
        summarize()
    elif args.mode == "worker":
        methods = METHODS[args.seed:] + METHODS[:args.seed]
        for scenario in ("clean", "noise"):
            for method in methods:
                if not (OUT / "cells" / ("%s_seed%d_%s.json" % (scenario, args.seed, method))).exists():
                    subprocess.run([sys.executable, str(Path(__file__).resolve()), "cell", "--seed", str(args.seed),
                        "--gpu", str(args.gpu), "--scenario", scenario, "--method", method], check=True)
        if not (OUT / "quality" / ("seed%d.json" % args.seed)).exists():
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "quality", "--seed", str(args.seed), "--gpu", str(args.gpu)], check=True)
    else:
        tests()
        if not (DATA / "manifest.json").exists():
            prepare()
        dump(OUT / "provenance.json", dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32, cuda=torch.version.cuda,
            methods=METHODS, seeds=[0, 1, 2], gpus=[0, 1, 2], n=1000, sequence_length=128,
            batch_size=8, learning_rate=LR, model="gpt2", trainable="transformer.ln_f",
            alpha=.2, k=10, gradient_gate=False, base_graph_score="Adam-aware FIRST order",
            sources={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (Path(__file__).resolve(), ROOT / "pile_main_100k/common.py", ROOT / "pile_main_100k/trajectory.py")},
            data_manifest_sha256=hashlib.sha256((DATA / "manifest.json").read_bytes()).hexdigest()))
        processes = []
        for seed in range(3):
            log = (OUT / ("seed%d.log" % seed)).open("a")
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "worker",
                "--seed", str(seed), "--gpu", str(seed)], stdout=log, stderr=subprocess.STDOUT)
            processes.append((process, log))
        codes = []
        for process, log in processes:
            codes.append(process.wait())
            log.close()
        if any(codes):
            raise RuntimeError("Workers failed: %s; inspect seed logs" % codes)
        summarize()


if __name__ == "__main__":
    main()
