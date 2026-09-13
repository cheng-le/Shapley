"""Shared protocol for the 10k-corpus Pile main table."""

from __future__ import annotations

import hashlib
import heapq
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from transformers import AutoTokenizer, GPT2LMHeadModel


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "Pile_large"
EXPERIMENT = ROOT / "pile_main_100k"
METHODS = [
    "Random",
    "BM25",
    "KNN-Shapley",
    "Influence Function",
    "TRAK (1 ckpt)",
    "LESS",
    "1st Order In-Run Data Shapley",
    "2nd Order In-Run Data Shapley",
    "Adam-aware 1st Order In-Run",
    "Adam-aware 2nd Order In-Run",
    "Graph Adam-aware 1st Order In-Run",
    "Graph Adam-aware 2nd Order In-Run",
]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_rows(split: str) -> list[dict]:
    rows = []
    with (DATA / f"{split}.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows.append({
                "text": str(row["text"]),
                "domain": str(row.get("meta", {}).get("pile_set_name", "Unknown")),
                "sha256": row.get("sha256", ""),
            })
    return rows


def stratified_sample_rows(rows: list[dict], limit: int, seed: int = 2027) -> list[dict]:
    """Select a deterministic domain-stratified utility subset."""
    if limit <= 0 or limit >= len(rows):
        return list(rows)
    grouped: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault(row["domain"], []).append(index)
    rng = np.random.default_rng(seed)
    for indices in grouped.values():
        rng.shuffle(indices)
    domains = sorted(grouped, key=lambda name: (-len(grouped[name]), name))
    chosen = [grouped[name].pop() for name in domains[:limit]]
    remaining = np.asarray([index for indices in grouped.values() for index in indices], dtype=np.int64)
    if len(chosen) < limit:
        chosen.extend(rng.choice(remaining, size=limit - len(chosen), replace=False).tolist())
    return [rows[int(index)] for index in chosen]


def stratified_split_indices(
    rows: list[dict], valuation_fraction: float = 0.2, seed: int = 2027
) -> tuple[np.ndarray, np.ndarray]:
    """Create disjoint, deterministic valuation and final-evaluation indices."""
    if not 0.0 < valuation_fraction < 1.0:
        raise ValueError("valuation_fraction must be between zero and one")
    grouped: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault(row["domain"], []).append(index)
    rng = np.random.default_rng(seed)
    valuation, evaluation = [], []
    for domain in sorted(grouped):
        indices = np.asarray(grouped[domain], dtype=np.int64)
        rng.shuffle(indices)
        count = min(len(indices) - 1, max(1, round(len(indices) * valuation_fraction))) if len(indices) > 1 else 0
        valuation.extend(indices[:count].tolist())
        evaluation.extend(indices[count:].tolist())
    return np.asarray(sorted(valuation), dtype=np.int64), np.asarray(sorted(evaluation), dtype=np.int64)


def dataset_fingerprint(rows: list[dict]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update((row.get("sha256") or hashlib.sha256(row["text"].encode()).hexdigest()).encode())
    return digest.hexdigest()[:16]


def corrupt_rows(rows: list[dict], fraction: float, seed: int) -> tuple[list[dict], np.ndarray]:
    rng = np.random.default_rng(seed)
    noisy = rng.choice(len(rows), size=round(len(rows) * fraction), replace=False)
    labels = np.zeros(len(rows), dtype=np.int8)
    labels[noisy] = 1
    output = [dict(row) for row in rows]
    # Replace a corrupted corpus by a length-matched token mixture from unrelated
    # corpora. This preserves superficial length statistics while breaking topic.
    donors = rng.permutation(len(rows))
    for position, index in enumerate(noisy):
        words = rows[index]["text"].split()
        donor_words = rows[int(donors[position])]["text"].split()
        target_len = max(8, min(len(words), 512))
        mixed = donor_words[: target_len // 2] + words[target_len // 2 : target_len]
        rng.shuffle(mixed)
        output[int(index)]["text"] = " ".join(mixed)
    return output, labels


def tokenizer():
    tok = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def model(device: torch.device, train_full: bool = False):
    net = GPT2LMHeadModel.from_pretrained(
        "gpt2", local_files_only=True, attn_implementation="eager"
    ).to(device)
    net.config.use_cache = False
    if not train_full:
        for parameter in net.parameters():
            parameter.requires_grad_(False)
        for parameter in net.transformer.ln_f.parameters():
            parameter.requires_grad_(True)
    return net


def encode(tok, texts: list[str], device: torch.device, length: int = 128):
    batch = tok(texts, padding="max_length", truncation=True, max_length=length, return_tensors="pt")
    return {key: value.to(device) for key, value in batch.items()}


def example_losses(net, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=batch["input_ids"].is_cuda):
        logits = net(**batch).logits[:, :-1].float()
    labels = batch["input_ids"][:, 1:]
    mask = batch["attention_mask"][:, 1:].float()
    losses = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="none")
    losses = losses.view(labels.shape)
    return (losses * mask).sum(1) / mask.sum(1).clamp_min(1)


def trainable(net):
    return tuple(parameter for parameter in net.parameters() if parameter.requires_grad)


def gradient_vector(loss: torch.Tensor, parameters, retain_graph: bool = False) -> torch.Tensor:
    gradients = torch.autograd.grad(loss, parameters, retain_graph=retain_graph)
    return torch.cat([gradient.flatten() for gradient in gradients]).float()


def query_gradient(net, tok, rows: list[dict], device: torch.device, limit: int = 64) -> np.ndarray:
    chosen = rows[:limit]
    net.zero_grad(set_to_none=True)
    loss = example_losses(net, encode(tok, [row["text"] for row in chosen], device)).mean()
    return gradient_vector(loss, trainable(net)).detach().cpu().numpy()


@torch.no_grad()
def embeddings(net, tok, rows: list[dict], device: torch.device, batch_size: int = 32) -> np.ndarray:
    net.eval()
    output = []
    for start in range(0, len(rows), batch_size):
        batch = encode(tok, [row["text"] for row in rows[start : start + batch_size]], device)
        hidden = net.transformer(**batch).last_hidden_state.float()
        mask = batch["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        output.append(F.normalize(pooled, dim=1).cpu().numpy().astype(np.float16))
    net.train()
    return np.concatenate(output)


def per_example_gradients(net, tok, rows: list[dict], device: torch.device, batch_size: int = 8) -> tuple[np.ndarray, np.ndarray]:
    parameters = trainable(net)
    all_gradients, all_losses = [], []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        losses = example_losses(net, encode(tok, [row["text"] for row in batch_rows], device))
        for index, loss in enumerate(losses):
            gradient = gradient_vector(loss, parameters, retain_graph=index + 1 < len(losses))
            all_gradients.append(gradient.detach().cpu().numpy().astype(np.float16))
        all_losses.extend(losses.detach().cpu().numpy().tolist())
        if (start // batch_size + 1) % 100 == 0:
            print(f"gradient rows={min(start + batch_size, len(rows)):,}/{len(rows):,}", flush=True)
    return np.stack(all_gradients), np.asarray(all_losses, dtype=np.float32)


def semantic_graph(features: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
    x = features.astype(np.float32)
    distances, indices = NearestNeighbors(n_neighbors=k + 1, metric="cosine", n_jobs=-1).fit(x).kneighbors(x)
    similarities = np.clip(1.0 - distances[:, 1:], 0.0, None)
    similarities /= similarities.sum(1, keepdims=True).clip(1e-8)
    return indices[:, 1:].astype(np.int32), similarities.astype(np.float32)


def graph_score(base: np.ndarray, gradients: np.ndarray, neighbors: np.ndarray, weights: np.ndarray) -> np.ndarray:
    g = gradients.astype(np.float32)
    g /= np.linalg.norm(g, axis=1, keepdims=True).clip(1e-8)
    agreement = np.maximum((g[:, None, :] * g[neighbors]).sum(2), 0.0)
    reliable = weights * agreement
    confidence = np.clip(reliable.sum(1), 0.0, 1.0)
    reliable /= reliable.sum(1, keepdims=True).clip(1e-8)
    support = (base[neighbors] * reliable).sum(1)
    alpha = 0.25 * confidence
    # Confidence gating blocks an isolated/noisy point from inheriting a large
    # value merely because one semantic neighbour is valuable.
    return (1.0 - alpha) * base + alpha * support


def confidence_gated_graph_weights(
    gradients: np.ndarray,
    neighbors: np.ndarray,
    weights: np.ndarray,
    chunk_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Gate semantic edges by positive gradient agreement without a dense graph."""
    norms = np.linalg.norm(gradients.astype(np.float32), axis=1).clip(1e-8)
    gated = np.empty_like(weights, dtype=np.float32)
    confidence = np.empty(len(neighbors), dtype=np.float32)
    for start in range(0, len(neighbors), chunk_size):
        stop = min(start + chunk_size, len(neighbors))
        source = gradients[start:stop].astype(np.float32)
        target = gradients[neighbors[start:stop]].astype(np.float32)
        agreement = np.maximum(
            (source[:, None, :] * target).sum(2)
            / (norms[start:stop, None] * norms[neighbors[start:stop]]),
            0.0,
        )
        reliable = weights[start:stop] * agreement
        row_confidence = np.clip(reliable.sum(1), 0.0, 1.0)
        gated[start:stop] = reliable / reliable.sum(1, keepdims=True).clip(1e-8)
        confidence[start:stop] = row_confidence
    return gated, confidence


def conservative_gradient_graph_scores(
    scores: np.ndarray,
    gradients: np.ndarray,
    neighbors: np.ndarray,
    weights: np.ndarray,
    smoothing: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Penalize unsupported high scores using gradient-compatible neighbors.

    This label-free correction mirrors the successful small-scale conservative
    graph method. It never promotes a point above its raw Adam-aware score.
    """
    gated_weights, confidence = confidence_gated_graph_weights(
        gradients, neighbors, weights
    )
    values = np.asarray(scores, dtype=np.float32)
    support = (values[neighbors] * gated_weights).sum(1)
    unsupported_excess = np.maximum(values - support, 0.0)
    corrected = values - smoothing * unsupported_excess
    return corrected.astype(np.float32), gated_weights, confidence


def full_graph_diffusion_scores(
    scores: np.ndarray,
    gradients: np.ndarray,
    neighbors: np.ndarray,
    weights: np.ndarray,
    alpha: float = 0.2,
    steps: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply stable full-vector diffusion instead of sparse one-pass replay."""
    gated_weights, confidence = confidence_gated_graph_weights(
        gradients, neighbors, weights
    )
    base = np.asarray(scores, dtype=np.float32)
    corrected = base.copy()

    def match_scale(term: np.ndarray) -> np.ndarray:
        term_scale = float(term.std())
        if term_scale < 1e-12:
            return np.zeros_like(term)
        return (term - float(term.mean())) * (float(base.std()) / term_scale)

    for _ in range(steps):
        support = (corrected[neighbors] * gated_weights).sum(1)
        corrected = base + alpha * confidence * match_scale(support)
    return corrected.astype(np.float32), gated_weights, confidence


def semantic_graph_diffusion_scores(
    scores: np.ndarray,
    neighbors: np.ndarray,
    weights: np.ndarray,
    alpha: float = 0.2,
    steps: int = 2,
) -> np.ndarray:
    """Diffuse complete scores over semantic KNN edges at matched scale."""
    base = np.asarray(scores, dtype=np.float32)
    corrected = base.copy()
    for _ in range(steps):
        support = (corrected[neighbors] * weights).sum(1)
        support_scale = float(support.std())
        if support_scale < 1e-12:
            break
        matched = (support - float(support.mean())) * (float(base.std()) / support_scale)
        corrected = base + alpha * matched
    return corrected.astype(np.float32)


def reciprocal_conservative_graph_scores(
    scores: np.ndarray,
    neighbors: np.ndarray,
    weights: np.ndarray,
    smoothing: float = 0.2,
    consistency_penalty: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Label-free Table-5 correction using reciprocal KNN as support."""
    values = np.asarray(scores, dtype=np.float64)
    source = np.arange(len(neighbors), dtype=neighbors.dtype)[:, None, None]
    reciprocal = (neighbors[neighbors] == source).any(axis=2)
    compatible_weights = weights.astype(np.float64) * reciprocal
    agreement = np.clip(compatible_weights.sum(1), 0.0, 1.0)
    support = np.divide(
        (compatible_weights * values[neighbors]).sum(1), agreement,
        out=values.copy(), where=agreement > 1e-12,
    )
    unsupported_excess = np.maximum(values - support, 0.0)
    corrected = (
        values - smoothing * unsupported_excess
        - consistency_penalty * float(values.std()) * (1.0 - agreement)
    )
    return corrected.astype(np.float32), agreement.astype(np.float32), reciprocal


def replay_graph_trajectory(
    raw_scores: np.ndarray,
    neighbors: np.ndarray,
    weights: np.ndarray,
    confidence: np.ndarray,
    seed: int,
    alpha: float = 0.2,
    batch_size: int = 8,
    include_pair: bool = False,
) -> np.ndarray:
    """Replay historical diffusion and the optional in-batch pair surrogate."""
    raw_scores = np.asarray(raw_scores, dtype=np.float32)
    history = np.zeros_like(raw_scores)
    corrected = np.zeros_like(raw_scores)
    order = np.random.default_rng(seed).permutation(len(raw_scores))

    def match_scale(term: np.ndarray, raw: np.ndarray) -> np.ndarray:
        raw_scale, term_scale = float(raw.std()), float(term.std())
        if raw_scale < 1e-12 or term_scale < 1e-12:
            return np.zeros_like(term)
        return (term - float(term.mean())) * (raw_scale / (term_scale + 1e-12))

    for start in range(0, len(order), batch_size):
        indices = order[start : start + batch_size]
        raw = raw_scores[indices]
        local_neighbors = neighbors[indices]
        local_weights = weights[indices]
        diffusion = (local_weights * history[local_neighbors]).sum(1)
        diffusion *= confidence[indices]
        value = raw + alpha * match_scale(diffusion, raw)
        if include_pair:
            positions = {int(index): position for position, index in enumerate(indices)}
            pair = np.zeros(len(indices), dtype=np.float32)
            for row, (row_neighbors, row_weights) in enumerate(zip(local_neighbors, local_weights)):
                for neighbor, weight in zip(row_neighbors, row_weights):
                    position = positions.get(int(neighbor))
                    if position is not None:
                        pair[row] += float(weight) * float(raw[position])
            pair *= confidence[indices]
            value += 0.5 * alpha * match_scale(pair, raw)
        corrected[indices] = value
        history[indices] += raw
    return corrected


def symmetric_graph(neighbors: np.ndarray, weights: np.ndarray):
    """Return the sparse affinity (W + W.T) / 2 used by the paper."""
    from scipy.sparse import coo_matrix

    count, degree = neighbors.shape
    rows = np.repeat(np.arange(count, dtype=np.int32), degree)
    directed = coo_matrix(
        (weights.reshape(-1), (rows, neighbors.reshape(-1))),
        shape=(count, count),
        dtype=np.float32,
    ).tocsr()
    return ((directed + directed.T) * 0.5).tocsr()


def diversity_select(
    scores: np.ndarray,
    neighbors: np.ndarray,
    weights: np.ndarray,
    count: int,
    lambda_fraction: float = 0.05,
) -> tuple[np.ndarray, dict[str, float]]:
    """Exact sparse greedy maximization of value minus graph redundancy."""
    values = np.asarray(scores, dtype=np.float64)
    graph = symmetric_graph(neighbors, weights)
    effective_lambda = float(lambda_fraction * values.std())
    redundancy = np.zeros(len(values), dtype=np.float64)
    versions = np.zeros(len(values), dtype=np.int32)
    chosen = np.zeros(len(values), dtype=bool)
    heap = [(-float(value), index, 0) for index, value in enumerate(values)]
    heapq.heapify(heap)
    selected = np.empty(count, dtype=np.int32)

    for position in range(count):
        while True:
            _, index, version = heapq.heappop(heap)
            if not chosen[index] and version == versions[index]:
                break
        selected[position] = index
        chosen[index] = True
        start, stop = graph.indptr[index], graph.indptr[index + 1]
        for candidate, affinity in zip(graph.indices[start:stop], graph.data[start:stop]):
            if chosen[candidate]:
                continue
            redundancy[candidate] += float(affinity)
            versions[candidate] += 1
            marginal = values[candidate] - effective_lambda * redundancy[candidate]
            heapq.heappush(heap, (-float(marginal), int(candidate), int(versions[candidate])))

    selected_mask = chosen
    selected_edge_mass = float(graph[selected_mask][:, selected_mask].sum()) * 0.5
    covered = selected_mask.copy()
    for index in selected:
        start, stop = graph.indptr[index], graph.indptr[index + 1]
        covered[graph.indices[start:stop]] = True
    diagnostics = {
        "lambda_fraction": float(lambda_fraction),
        "effective_lambda": effective_lambda,
        "selected_pair_affinity": selected_edge_mass,
        "selected_pair_affinity_per_item": selected_edge_mass / max(count, 1),
        "graph_coverage": float(covered.mean()),
    }
    return selected, diagnostics


def rank_scores(gradients: np.ndarray, features: np.ndarray, query_grad: np.ndarray, query_embedding: np.ndarray, seed: int, graph=None) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    g = gradients.astype(np.float32)
    q = query_grad.astype(np.float32)
    first = g @ q
    diagonal = np.square(q) + np.mean(np.square(g), axis=0)
    second = first - 0.5 * (np.square(g) @ diagonal) / max(len(g), 1)
    preconditioner = 1.0 / (np.sqrt(np.mean(np.square(g), axis=0)) + 1e-6)
    adam_first = (g * preconditioner) @ q
    adam_second = adam_first - 0.5 * (np.square(g * preconditioner) @ diagonal) / max(len(g), 1)
    neighbors, weights = graph if graph is not None else semantic_graph(features)
    semantic = features.astype(np.float32) @ query_embedding.astype(np.float32)
    scores = {
        "Random": rng.standard_normal(len(g)),
        "BM25": semantic,  # Filled with lexical BM25 by score runner when available.
        "KNN-Shapley": semantic,
        "Influence Function": first,
        "TRAK (1 ckpt)": first / (np.linalg.norm(g, axis=1).clip(1e-8)),
        "LESS": first / (np.linalg.norm(g, axis=1).clip(1e-8) * np.linalg.norm(q).clip(1e-8)),
        "1st Order In-Run Data Shapley": first,
        "2nd Order In-Run Data Shapley": second,
        "Adam-aware 1st Order In-Run": adam_first,
        "Adam-aware 2nd Order In-Run": adam_second,
        "Graph Adam-aware 1st Order In-Run": graph_score(adam_first, g, neighbors, weights),
        "Graph Adam-aware 2nd Order In-Run": graph_score(adam_second, g, neighbors, weights),
    }
    return {name: np.asarray(scores[name], dtype=np.float32) for name in METHODS}


def domain_counts(rows: list[dict]) -> dict[str, int]:
    return dict(Counter(row["domain"] for row in rows))
