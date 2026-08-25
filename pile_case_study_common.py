"""Shared GPT-2 utilities for the paper's Pile case-study reproductions."""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from transformers import AutoTokenizer, GPT2LMHeadModel


ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class LocalPileProtocol:
    model_name: str = "gpt2"
    data_path: str = "data/Pile_large/train.jsonl"
    sequence_length: int = 128
    batch_size: int = 4
    learning_rate: float = 6e-4
    min_learning_rate: float = 6e-5
    warmup_steps: int = 20
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    graph_k: int = 10
    graph_alpha: float = 0.2
    graph_lambda: float = 0.05
    seed: int = 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl_records(path: str | Path, limit: int = 0) -> list[dict]:
    records = []
    with Path(path).open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            row = json.loads(line)
            text = str(row.get("text", "")).strip()
            if text:
                records.append({"text": text, "domain": row.get("meta", {}).get("pile_set_name", "Unknown")})
            if limit and len(records) >= limit:
                break
    return records


def load_domain_records(subset_dir: str | Path, per_domain: int) -> list[dict]:
    records = []
    for path in sorted(Path(subset_dir).glob("*.jsonl")):
        rows = load_jsonl_records(path, per_domain)
        records.extend(rows)
    return records


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_attribution_model(protocol: LocalPileProtocol):
    model = GPT2LMHeadModel.from_pretrained(
        protocol.model_name,
        local_files_only=True,
        attn_implementation="eager",
    ).to(DEVICE)
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.transformer.ln_f.parameters():
        parameter.requires_grad_(True)
    model.train()
    return model


def tokenize_texts(tokenizer, texts: list[str], sequence_length: int) -> dict[str, torch.Tensor]:
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=sequence_length,
        return_tensors="pt",
    )


def per_example_losses(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        shift_logits = logits[:, :-1].float()
        shift_labels = input_ids[:, 1:]
        shift_mask = attention_mask[:, 1:].float()
        token_losses = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="none",
        ).view(input_ids.size(0), -1)
        return (token_losses * shift_mask).sum(1) / shift_mask.sum(1).clamp_min(1.0)


def trainable_parameters(model) -> tuple[torch.nn.Parameter, ...]:
    return tuple(model.transformer.ln_f.parameters())


def flat_gradient(loss, parameters, retain_graph: bool) -> torch.Tensor:
    gradients = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, create_graph=False)
    return torch.cat([gradient.reshape(-1) for gradient in gradients]).float()


def batch_gradients(model, encoded: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    ids = encoded["input_ids"].to(DEVICE)
    mask = encoded["attention_mask"].to(DEVICE)
    losses = per_example_losses(model, ids, mask)
    parameters = trainable_parameters(model)
    gradients = []
    for index, loss in enumerate(losses):
        gradients.append(flat_gradient(loss, parameters, retain_graph=index + 1 < len(losses)))
    return torch.stack(gradients), losses.detach()


@torch.no_grad()
def semantic_embeddings(model, tokenizer, texts: list[str], sequence_length: int, batch_size: int = 16) -> np.ndarray:
    model.eval()
    chunks = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenize_texts(tokenizer, texts[start : start + batch_size], sequence_length)
        ids = encoded["input_ids"].to(DEVICE)
        mask = encoded["attention_mask"].to(DEVICE)
        hidden = model.transformer(input_ids=ids, attention_mask=mask).last_hidden_state.float()
        pooled = (hidden * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        chunks.append(F.normalize(pooled, dim=1).cpu())
    model.train()
    return torch.cat(chunks).numpy()


def build_semantic_graph(embeddings: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    count = min(k + 1, len(embeddings))
    distances, indices = NearestNeighbors(n_neighbors=count, metric="cosine").fit(embeddings).kneighbors(embeddings)
    weights = np.clip(1.0 - distances[:, 1:], 0.0, None)
    weights /= np.maximum(weights.sum(1, keepdims=True), 1e-12)
    return indices[:, 1:].astype(np.int64), weights.astype(np.float64)


def graph_regularize(scores: np.ndarray, neighbors: np.ndarray, weights: np.ndarray, alpha: float, penalty: float) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    support = (scores[..., neighbors] * weights).sum(axis=-1)
    residual = scores - support
    scale = np.std(scores, axis=-1, keepdims=True) + 1e-12
    disagreement = np.abs(residual) / scale
    return (1.0 - alpha) * scores + alpha * support - penalty * scale * np.tanh(disagreement)


def learning_rate_at(step: int, total_steps: int, protocol: LocalPileProtocol) -> float:
    if step < protocol.warmup_steps:
        return protocol.learning_rate * (step + 1) / max(protocol.warmup_steps, 1)
    progress = (step - protocol.warmup_steps) / max(total_steps - protocol.warmup_steps, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return protocol.min_learning_rate + cosine * (protocol.learning_rate - protocol.min_learning_rate)


def make_optimizer(model, protocol: LocalPileProtocol):
    return torch.optim.AdamW(
        trainable_parameters(model),
        lr=protocol.learning_rate,
        betas=(protocol.beta1, protocol.beta2),
        weight_decay=protocol.weight_decay,
    )


def update_model(model, optimizer, encoded, lr: float, grad_clip: float) -> float:
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.zero_grad(set_to_none=True)
    losses = per_example_losses(model, encoded["input_ids"].to(DEVICE), encoded["attention_mask"].to(DEVICE))
    loss = losses.mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable_parameters(model), grad_clip)
    optimizer.step()
    return float(loss.detach())


def bm25_scores(documents: list[str], queries: list[str], k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    tokenized = [[token.lower() for token in text.split()] for text in documents]
    doc_freq = Counter()
    for document in tokenized:
        doc_freq.update(set(document))
    n_docs = len(documents)
    avg_len = np.mean([len(document) for document in tokenized])
    term_counts = [Counter(document) for document in tokenized]
    output = np.zeros((len(queries), n_docs), dtype=np.float64)
    for query_index, query in enumerate(queries):
        for term in set(token.lower() for token in query.split()):
            frequency = doc_freq.get(term, 0)
            idf = math.log(1.0 + (n_docs - frequency + 0.5) / (frequency + 0.5))
            for doc_index, counts in enumerate(term_counts):
                tf = counts.get(term, 0)
                denominator = tf + k1 * (1.0 - b + b * len(tokenized[doc_index]) / max(avg_len, 1.0))
                if denominator:
                    output[query_index, doc_index] += idf * tf * (k1 + 1.0) / denominator
    return output


def protocol_metadata(protocol: LocalPileProtocol) -> dict:
    return {
        "paper_target": {
            "model": "GPT2 124M",
            "dataset": "uniform random Pile subset, about 10B tokens",
            "sequence_length": 1024,
            "batch_size": 16,
            "max_learning_rate": 6e-4,
            "min_learning_rate": 6e-5,
            "optimizer": "AdamW",
            "weight_decay": 0.1,
            "betas": [0.9, 0.95],
            "gradient_clip": 1.0,
            "warmup_steps": 2000,
            "training_steps": 500000,
        },
        "local_executable": asdict(protocol),
        "approximation": "GPT2 backbone frozen; attribution and updates use final LayerNorm's 1,536 parameters.",
    }


def save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
