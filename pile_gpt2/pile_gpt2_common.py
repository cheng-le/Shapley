"""Pile + GPT2 版 In-Run Data Shapley 公共代码。

这个分支复现论文主实验的语言模型场景：训练样本是文本块，模型是 GPT2
因果语言模型。通用 Shapley 公式、MC fidelity、drop/keep 和总图仍复用
根目录 shapley_common.py；本文件只处理文本数据、GPT2 模型、LM loss、token
accuracy 和文本样本可视化。
"""

from __future__ import annotations

import argparse
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Dataset, Subset

import shapley_common as core
from shapley_common import (
    DEVICE,
    IndexedDataset,
    LocalStepSnapshot,
    add_dicts,
    adam_update_direction_from_grad,
    clone_adam_state,
    compute_single_grad,
    compute_val_grad_and_loss,
    dot_dict,
    finish_experiment,
    finish_graph_comparison_experiment,
    hvp_from_val_grad,
    sample_validation_batch,
    save_scores,
    scale_dict,
    set_seed,
    summarize_scores,
    TopologyGraph,
    topology_regularize_scores,
    topology_second_order_surrogate_scores,
)


@dataclass
class PileGPT2RunConfig:
    seed: int
    batch_size: int
    epochs_score: int
    lr: float
    hidden_dim: int
    train_limit: int
    val_limit: int
    val_samples: int
    output: str
    model_name: str = "gpt2"
    dataset_name: str = "EleutherAI/pile"
    dataset_config: str = "all"
    train_split: str = "train"
    val_split: str = "validation"
    text_field: str = "text"
    dataset_format: str = "hf"
    allow_remote_pile_download: bool = False
    local_train_text: str = ""
    local_val_text: str = ""
    block_size: int = 64
    stride: int = 64
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.01
    topo_k: int = 10
    topo_alpha: float = 0.2
    topo_lambda: float = 0.05
    topo_feature_dim: int = 2048
    plot_output: str = ""
    mc_batch_size: int = 2
    mc_permutations: int = 5
    prune_epochs: int = 1
    snapshot_after_batches: int = 0
    curvature_clip_ratio: float = 0.25


def require_transformers():
    try:
        from transformers import AutoTokenizer, GPT2LMHeadModel
    except ImportError as exc:
        raise RuntimeError(
            "Pile+GPT2 实验需要 transformers。请先安装：\n"
            "/home/lcheng/anaconda3/envs/torch_5090/bin/python -m pip install transformers datasets accelerate"
        ) from exc
    return AutoTokenizer, GPT2LMHeadModel


def maybe_load_hf_dataset(config: PileGPT2RunConfig, split: str) -> List[str]:
    if config.dataset_name == "EleutherAI/pile" and not config.allow_remote_pile_download:
        raise RuntimeError(
            "默认不直接从 EleutherAI/pile 下载数据：旧加载脚本会访问 the-eye.eu 的 30 个大分片，"
            "当前还可能遇到 SSL 证书过期。\n"
            "建议做法：手动准备 Pile 的 jsonl/parquet 小分片，然后使用：\n"
            "  --local-train-text data/Pile/train.jsonl --local-val-text data/Pile/val.jsonl "
            "--dataset-format jsonl --text-field text\n"
            "如果你确认要走旧脚本远程下载，请显式添加 --allow-remote-pile-download。"
        )
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "没有提供本地文本文件时，需要 datasets 来读取 Pile。请先安装 datasets，"
            "或使用 --local-train-text/--local-val-text 指向本地 .txt 文件。"
        ) from exc

    try:
        ds = load_dataset(config.dataset_name, config.dataset_config, split=split, streaming=False, trust_remote_code=config.allow_remote_pile_download)
    except (RuntimeError, FileNotFoundError, OSError) as exc:
        msg = str(exc)
        if "Dataset scripts are no longer supported" in msg:
            raise RuntimeError(
                "当前 datasets 版本不再支持 EleutherAI/pile 仓库里的 pile.py 加载脚本。\n"
                "建议手动准备 Pile 的 jsonl/parquet 分片，然后使用 --local-train-text/--local-val-text "
                "和 --dataset-format jsonl/parquet 读取本地文件。"
            ) from exc
        if "the-eye.eu" in msg or "CERTIFICATE_VERIFY_FAILED" in msg or "00.jsonl.zst" in msg:
            raise RuntimeError(
                "EleutherAI/pile 旧脚本正在从 the-eye.eu 下载原始 Pile 分片，但该源当前 SSL/可用性不稳定，"
                "而且会尝试准备很大的分片。\n"
                "请手动下载或整理一个 Pile 子集到本地 jsonl/parquet，再使用 --local-train-text/--local-val-text "
                "和 --dataset-format jsonl/parquet 运行。"
            ) from exc
        raise
    texts: List[str] = []
    limit = config.train_limit if split == config.train_split else config.val_limit
    for row in ds:
        text = str(row.get(config.text_field, "")).strip()
        if text:
            texts.append(text)
        if 0 < limit <= len(texts):
            break
    return texts


def read_local_text(path: str, limit: int) -> List[str]:
    rows: List[str] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            text = line.strip()
            if text:
                rows.append(text)
            if 0 < limit <= len(rows):
                break
    return rows


def read_local_records(path: str, limit: int, text_field: str, dataset_format: str) -> List[str]:
    """读取本地文本/JSON/Parquet 分片；用于绕开新版 datasets 禁用 Pile 脚本的问题。"""
    fmt = dataset_format.lower()
    if fmt in {"", "hf", "text", "txt"}:
        return read_local_text(path, limit)
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("读取 json/parquet/csv 本地数据需要安装 datasets。") from exc
    if fmt == "jsonl":
        fmt = "json"
    ds = load_dataset(fmt, data_files=path, split="train")
    rows: List[str] = []
    for row in ds:
        text = str(row.get(text_field, "")).strip()
        if text:
            rows.append(text)
        if 0 < limit <= len(rows):
            break
    return rows


class TextBlockDataset(Dataset):
    """把文本切成固定长度 token block，每个 block 作为一个训练样本。"""

    def __init__(self, texts: List[str], tokenizer, block_size: int, limit: int, stride: int | None = None):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.pad_id = int(tokenizer.pad_token_id)
        step = max(1, int(stride or block_size))
        blocks: List[torch.Tensor] = []
        max_token_len = min(int(getattr(tokenizer, "model_max_length", 1024) or 1024), max(block_size, block_size + step * max(limit, 1)))
        for text in texts:
            encoded = tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_token_len)
            ids = encoded["input_ids"]
            if len(ids) < 2:
                continue
            for start in range(0, max(1, len(ids) - 1), step):
                chunk = ids[start : start + block_size]
                if len(chunk) < 2:
                    continue
                if len(chunk) < block_size:
                    chunk = chunk + [self.pad_id] * (block_size - len(chunk))
                blocks.append(torch.tensor(chunk, dtype=torch.long))
                if 0 < limit <= len(blocks):
                    break
            if 0 < limit <= len(blocks):
                break
        if not blocks:
            raise RuntimeError("没有构造出任何文本 block，请检查文本文件、tokenizer 或 block_size。")
        self.input_ids = torch.stack(blocks)
        labels = self.input_ids.clone()
        labels[labels == self.pad_id] = -100
        self.labels = labels
        self.targets = [0 for _ in range(len(self.input_ids))]

    def __len__(self) -> int:
        return int(self.input_ids.size(0))

    def __getitem__(self, idx: int):
        return self.input_ids[idx], self.labels[idx]


_GPT2_CACHE_NAME = None
_GPT2_CACHE_CONFIG = None
_GPT2_CACHE_STATE = None


class GPT2LogitsWrapper(nn.Module):
    """把 HuggingFace GPT2 包成只返回 logits 的 nn.Module，适配通用 loss_fn(model(x), y)。"""

    def __init__(self, model_name: str):
        super().__init__()
        global _GPT2_CACHE_NAME, _GPT2_CACHE_CONFIG, _GPT2_CACHE_STATE
        _, GPT2LMHeadModel = require_transformers()
        if _GPT2_CACHE_NAME == model_name and _GPT2_CACHE_CONFIG is not None and _GPT2_CACHE_STATE is not None:
            _GPT2_CACHE_CONFIG._attn_implementation = "eager"
            self.model = GPT2LMHeadModel(_GPT2_CACHE_CONFIG)
            self.model.load_state_dict({k: v.clone() for k, v in _GPT2_CACHE_STATE.items()})
        else:
            self.model = GPT2LMHeadModel.from_pretrained(model_name, attn_implementation="eager")
            self.model.config._attn_implementation = "eager"
            _GPT2_CACHE_NAME = model_name
            _GPT2_CACHE_CONFIG = self.model.config
            _GPT2_CACHE_STATE = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        self.model.config.use_cache = False

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids).logits


class CausalLMLoss(nn.Module):
    """GPT2 因果语言模型 loss：预测下一个 token。"""

    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        return self.ce(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


def evaluate_lm(model: nn.Module, loader: DataLoader, loss_fn: nn.Module) -> Tuple[float, float]:
    """返回平均 token loss 和 token accuracy，供通用 drop/keep 图使用。"""
    model.eval()
    total_loss, total_correct, total_count = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            for start in range(0, x.size(0), 8):
                xb = x[start : start + 8].to(DEVICE)
                yb = y[start : start + 8].to(DEVICE)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                shift_logits = logits[:, :-1, :]
                shift_labels = yb[:, 1:]
                mask = shift_labels != -100
                pred = shift_logits.argmax(dim=-1)
                total_correct += ((pred == shift_labels) & mask).sum().item()
                token_count = int(mask.sum().item())
                total_count += token_count
                total_loss += loss.item() * max(token_count, 1)
    return total_loss / max(total_count, 1), total_correct / max(total_count, 1)


_ACTIVE_CONFIG: PileGPT2RunConfig | None = None
_TOKENIZER = None


def get_tokenizer(config: PileGPT2RunConfig):
    global _TOKENIZER
    if _TOKENIZER is None:
        AutoTokenizer, _ = require_transformers()
        tok = AutoTokenizer.from_pretrained(config.model_name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        _TOKENIZER = tok
    return _TOKENIZER


def make_pile_model(hidden_dim: int) -> nn.Module:
    if _ACTIVE_CONFIG is None:
        raise RuntimeError("尚未安装 Pile+GPT2 hooks。")
    return GPT2LogitsWrapper(_ACTIVE_CONFIG.model_name).to(DEVICE)


def make_pile_optimizer(model: nn.Module, optimizer_name: str, config: PileGPT2RunConfig) -> optim.Optimizer:
    if optimizer_name == "sgd":
        return optim.SGD(model.parameters(), lr=config.lr)
    return optim.AdamW(model.parameters(), lr=config.lr, betas=(config.beta1, config.beta2), eps=config.eps, weight_decay=config.weight_decay)


def install_pile_hooks(config: PileGPT2RunConfig) -> None:
    global _ACTIVE_CONFIG
    _ACTIVE_CONFIG = config
    core.configure_runtime(
        model_factory=make_pile_model,
        optimizer_factory=make_pile_optimizer,
        loss_factory=lambda cfg: CausalLMLoss(),
        sample_plotter=plot_text_sample_grid,
        utility_train_mode=True,
        evaluator=evaluate_lm,
    )


def load_texts(config: PileGPT2RunConfig) -> Tuple[List[str], List[str]]:
    if config.local_train_text:
        train_texts = read_local_records(config.local_train_text, config.train_limit, config.text_field, config.dataset_format)
    else:
        train_texts = maybe_load_hf_dataset(config, config.train_split)
    if config.local_val_text:
        val_texts = read_local_records(config.local_val_text, config.val_limit, config.text_field, config.dataset_format)
    else:
        val_texts = maybe_load_hf_dataset(config, config.val_split)
    return train_texts, val_texts


def make_pile_experiment_loaders(config: PileGPT2RunConfig):
    install_pile_hooks(config)
    tokenizer = get_tokenizer(config)
    train_texts, val_texts = load_texts(config)
    base_train = TextBlockDataset(train_texts, tokenizer, config.block_size, config.train_limit, config.stride)
    val_dataset = TextBlockDataset(val_texts, tokenizer, config.block_size, config.val_limit, config.stride)
    train_loader = DataLoader(IndexedDataset(base_train), batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=max(1, min(16, config.batch_size)), shuffle=False)
    return base_train, val_dataset, train_loader, val_loader


def plot_text_sample_grid(ax, dataset: Dataset, sample_indices: np.ndarray, title: str, scores: np.ndarray) -> None:
    tokenizer = get_tokenizer(_ACTIVE_CONFIG)
    ax.set_title(title, fontsize=10, pad=6)
    ax.axis("off")
    rows = []
    for rank, idx in enumerate(sample_indices[:10], start=1):
        x, _ = dataset[int(idx)]
        text = tokenizer.decode([int(t) for t in x.tolist() if int(t) != tokenizer.pad_token_id])
        text = " ".join(text.replace("\n", " ").split())
        snippet = textwrap.shorten(text, width=86, placeholder="...")
        snippet = snippet.replace("$", "\\$")
        rows.append(f"{rank:02d}. i={int(idx)}  s={scores[int(idx)]:.3g}  {snippet}")
    ax.text(0.01, 0.98, "\n\n".join(rows), va="top", ha="left", fontsize=7, family="monospace", wrap=True)


def add_pile_args(parser: argparse.ArgumentParser, optimizer_name: str) -> None:
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs-score", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4 if optimizer_name == "sgd" else 5e-5)
    parser.add_argument("--hidden-dim", type=int, default=0, help="兼容通用接口；GPT2 不使用该参数。")
    parser.add_argument("--train-limit", type=int, default=64)
    parser.add_argument("--val-limit", type=int, default=64)
    parser.add_argument("--val-samples", type=int, default=4)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--plot-output", type=str, default="")
    parser.add_argument("--model-name", type=str, default="gpt2")
    parser.add_argument("--dataset-name", type=str, default="EleutherAI/pile")
    parser.add_argument("--dataset-config", type=str, default="all")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="validation")
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument("--dataset-format", type=str, default="hf", choices=["hf", "text", "txt", "json", "jsonl", "parquet", "csv"], help="本地数据格式；读取 EleutherAI/pile 时保持 hf，手动下载分片时用 json/jsonl/parquet。")
    parser.add_argument("--allow-remote-pile-download", action="store_true", help="显式允许 EleutherAI/pile 旧脚本从远程源下载大分片；默认关闭以避免 the-eye.eu SSL/大文件问题。")
    parser.add_argument("--local-train-text", type=str, default="", help="本地训练文本，每行一个样本；设置后不读取 HuggingFace Pile。")
    parser.add_argument("--local-val-text", type=str, default="", help="本地验证文本，每行一个样本；设置后不读取 HuggingFace Pile。")
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--mc-batch-size", type=int, default=2)
    parser.add_argument("--mc-permutations", type=int, default=5)
    parser.add_argument("--prune-epochs", type=int, default=1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--snapshot-after-batches", type=int, default=0)
    parser.add_argument("--curvature-clip-ratio", type=float, default=0.25)
    if optimizer_name in {"adam", "adam_graph"}:
        parser.add_argument("--beta1", type=float, default=0.9)
        parser.add_argument("--beta2", type=float, default=0.999)
        parser.add_argument("--eps", type=float, default=1e-8)
    if optimizer_name == "adam_graph":
        parser.add_argument("--topo-k", type=int, default=10)
        parser.add_argument("--topo-alpha", type=float, default=0.2)
        parser.add_argument("--topo-lambda", type=float, default=0.05)
        parser.add_argument("--topo-feature-dim", type=int, default=2048)


def pile_config_from_args(args: argparse.Namespace) -> PileGPT2RunConfig:
    return PileGPT2RunConfig(**vars(args))


def snapshot_target_batch(config: PileGPT2RunConfig, train_loader: DataLoader) -> int:
    total_batches = max(1, len(train_loader) * config.epochs_score)
    return min(max(0, config.snapshot_after_batches), total_batches - 1)


def enter_score_mode(model: nn.Module) -> bool:
    """打分时关闭 GPT2 dropout，使 per-sample 梯度可复现。"""
    was_training = model.training
    model.eval()
    return was_training


def restore_model_mode(model: nn.Module, was_training: bool) -> None:
    model.train(was_training)


def build_knn_graph_from_token_blocks(dataset: TextBlockDataset, k: int, feature_dim: int) -> TopologyGraph:
    """用 token 哈希直方图建 KNN 图，作为文本样本的轻量拓扑近似。"""
    n = len(dataset)
    rows, cols, vals = [], [], []
    for i in range(n):
        ids = dataset.input_ids[i].numpy()
        ids = ids[ids != dataset.pad_id]
        if ids.size == 0:
            continue
        bins = ids % feature_dim
        uniq, counts = np.unique(bins, return_counts=True)
        norm = float(np.linalg.norm(counts)) + 1e-12
        rows.extend([i] * len(uniq))
        cols.extend(uniq.tolist())
        vals.extend((counts / norm).tolist())
    x = sp.csr_matrix((vals, (rows, cols)), shape=(n, feature_dim), dtype=np.float32)
    n_neighbors = min(k + 1, n)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine").fit(x)
    distances, indices = nbrs.kneighbors(x)
    neighbors = indices[:, 1:].astype(np.int64, copy=True)
    raw_weights = np.clip(1.0 - distances[:, 1:], 0.0, None).astype(np.float32, copy=True)
    density = raw_weights.mean(axis=1).astype(np.float32)
    row_sum = raw_weights.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0.0] = 1.0
    weights = raw_weights / row_sum
    return TopologyGraph(neighbors=neighbors, weights=weights, density=density)


def combine_first_and_curvature(first_terms: List[float], curvature_terms: List[float], config: PileGPT2RunConfig) -> List[float]:
    """语言模型 HVP 噪声更大，用信任域缩放防止曲率项压过一阶项。"""
    first = np.asarray(first_terms, dtype=np.float64)
    curvature = np.asarray(curvature_terms, dtype=np.float64)
    ratio = float(config.curvature_clip_ratio)
    if ratio >= 0.0:
        first_scale = float(np.std(first))
        curvature_scale = float(np.std(curvature))
        max_scale = ratio * first_scale
        if curvature_scale > max_scale and curvature_scale > 1e-12:
            curvature = curvature * (max_scale / (curvature_scale + 1e-12))
    return (first + curvature).astype(np.float64).tolist()


def compute_sgd_first_order(config: PileGPT2RunConfig):
    """一阶 SGD：phi_i = -(lr / B) * <g_val, g_i>。"""
    set_seed(config.seed)
    base_train, val_dataset, train_loader, val_loader = make_pile_experiment_loaders(config)
    model = make_pile_model(config.hidden_dim)
    optimizer = optim.SGD(model.parameters(), lr=config.lr)
    loss_fn = CausalLMLoss()
    scores = torch.zeros(len(train_loader.dataset), dtype=torch.float32)
    val_imgs, val_labels = sample_validation_batch(val_loader, config.val_samples)
    snapshot, snapshot_target, scored_batches = None, snapshot_target_batch(config, train_loader), 0
    score_start = time.perf_counter()

    for _ in range(config.epochs_score):
        model.train()
        for images, labels, indices in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            state_before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            was_training = enter_score_mode(model)
            _, val_grad = compute_val_grad_and_loss(model, loss_fn, val_imgs, val_labels, create_graph=False)
            lr_scale = -optimizer.param_groups[0]["lr"] / images.size(0)
            local_scores = []
            for i in range(images.size(0)):
                grad_i = compute_single_grad(model, loss_fn, images[i], labels[i])
                score_i = lr_scale * dot_dict(val_grad, grad_i)
                local_scores.append(score_i)
                scores[int(indices[i])] += score_i
            if snapshot is None and scored_batches >= snapshot_target:
                snapshot = LocalStepSnapshot(state_before, images.detach().cpu(), labels.detach().cpu(), indices.detach().cpu(), val_imgs.detach().cpu(), val_labels.detach().cpu(), np.asarray(local_scores), optimizer.param_groups[0]["lr"], images.size(0), "sgd")
            restore_model_mode(model, was_training)
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            scored_batches += 1
    return base_train, val_dataset, scores, snapshot, {"Score time": time.perf_counter() - score_start}


def compute_adam_first_order(config: PileGPT2RunConfig):
    """Adam-aware 一阶：固定 Adam 状态后，用单样本梯度进入 Adam 更新式。"""
    set_seed(config.seed)
    base_train, val_dataset, train_loader, val_loader = make_pile_experiment_loaders(config)
    model = make_pile_model(config.hidden_dim)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, betas=(config.beta1, config.beta2), eps=config.eps, weight_decay=config.weight_decay)
    loss_fn = CausalLMLoss()
    scores = torch.zeros(len(train_loader.dataset), dtype=torch.float32)
    val_imgs, val_labels = sample_validation_batch(val_loader, config.val_samples)
    snapshot, snapshot_target, scored_batches = None, snapshot_target_batch(config, train_loader), 0
    score_start = time.perf_counter()

    for _ in range(config.epochs_score):
        model.train()
        for images, labels, indices in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            state_before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            adam_state = clone_adam_state(model, optimizer)
            was_training = enter_score_mode(model)
            _, val_grad = compute_val_grad_and_loss(model, loss_fn, val_imgs, val_labels, create_graph=False)
            lr_now = optimizer.param_groups[0]["lr"]
            local_scores = []
            for i in range(images.size(0)):
                grad_i = compute_single_grad(model, loss_fn, images[i], labels[i])
                update_i = adam_update_direction_from_grad(grad_i, adam_state, config.beta1, config.beta2, config.eps)
                score_i = -lr_now * dot_dict(val_grad, update_i)
                local_scores.append(score_i)
                scores[int(indices[i])] += score_i
            if snapshot is None and scored_batches >= snapshot_target:
                snap_state = {k: {"exp_avg": v["exp_avg"].detach().cpu().clone(), "exp_avg_sq": v["exp_avg_sq"].detach().cpu().clone(), "step": int(v["step"])} for k, v in adam_state.items()}
                snapshot = LocalStepSnapshot(state_before, images.detach().cpu(), labels.detach().cpu(), indices.detach().cpu(), val_imgs.detach().cpu(), val_labels.detach().cpu(), np.asarray(local_scores), lr_now, images.size(0), "adam", adam_state=snap_state)
            restore_model_mode(model, was_training)
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            scored_batches += 1
    return base_train, val_dataset, scores, snapshot, {"Score time": time.perf_counter() - score_start}


def compute_adam_graph_first_order(config: PileGPT2RunConfig):
    """Adam-aware 一阶 + token KNN 图扩散稳定化。"""
    set_seed(config.seed)
    base_train, val_dataset, train_loader, val_loader = make_pile_experiment_loaders(config)
    graph_start = time.perf_counter()
    graph = build_knn_graph_from_token_blocks(base_train, config.topo_k, config.topo_feature_dim)
    graph_time = time.perf_counter() - graph_start
    model = make_pile_model(config.hidden_dim)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, betas=(config.beta1, config.beta2), eps=config.eps, weight_decay=config.weight_decay)
    loss_fn = CausalLMLoss()
    raw_scores = torch.zeros(len(train_loader.dataset), dtype=torch.float32)
    topo_scores = torch.zeros(len(train_loader.dataset), dtype=torch.float32)
    val_imgs, val_labels = sample_validation_batch(val_loader, config.val_samples)
    snapshot, snapshot_target, scored_batches = None, snapshot_target_batch(config, train_loader), 0
    score_start = time.perf_counter(); raw_score_time = 0.0; topology_time = 0.0

    for _ in range(config.epochs_score):
        model.train()
        for images, labels, indices in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            state_before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            adam_state = clone_adam_state(model, optimizer)
            was_training = enter_score_mode(model)
            _, val_grad = compute_val_grad_and_loss(model, loss_fn, val_imgs, val_labels, create_graph=False)
            lr_now = optimizer.param_groups[0]["lr"]
            raw_start = time.perf_counter(); raw_batch = []
            for i in range(images.size(0)):
                grad_i = compute_single_grad(model, loss_fn, images[i], labels[i])
                update_i = adam_update_direction_from_grad(grad_i, adam_state, config.beta1, config.beta2, config.eps)
                raw_batch.append(-lr_now * dot_dict(val_grad, update_i))
            raw_score_time += time.perf_counter() - raw_start
            batch_idx = indices.numpy(); raw_np = np.asarray(raw_batch, dtype=np.float32)
            topo_start = time.perf_counter()
            topo_np = topology_regularize_scores(batch_idx, raw_np, raw_scores.numpy(), graph, config.topo_alpha, config.topo_lambda)
            topology_time += time.perf_counter() - topo_start
            raw_scores[batch_idx] += torch.from_numpy(raw_np)
            topo_scores[batch_idx] += torch.from_numpy(topo_np.astype(np.float32))
            if snapshot is None and scored_batches >= snapshot_target:
                snap_state = {k: {"exp_avg": v["exp_avg"].detach().cpu().clone(), "exp_avg_sq": v["exp_avg_sq"].detach().cpu().clone(), "step": int(v["step"])} for k, v in adam_state.items()}
                snapshot = LocalStepSnapshot(state_before, images.detach().cpu(), labels.detach().cpu(), indices.detach().cpu(), val_imgs.detach().cpu(), val_labels.detach().cpu(), np.asarray(raw_batch), lr_now, images.size(0), "adam", adam_state=snap_state, topo_approx_scores=np.asarray(topo_np))
            restore_model_mode(model, was_training)
            optimizer.zero_grad(); loss = loss_fn(model(images), labels); loss.backward(); optimizer.step(); scored_batches += 1
    score_time = time.perf_counter() - score_start
    timing = {"Graph build time": graph_time, "Score time": score_time, "Raw Adam score time": raw_score_time, "Raw first-order time": raw_score_time, "Topology distance time": topology_time, "Train update time": max(0.0, score_time - raw_score_time - topology_time), "Total graph+score time": graph_time + score_time}
    return base_train, val_dataset, raw_scores, topo_scores, snapshot, timing


def compute_sgd_second_order(config: PileGPT2RunConfig):
    """二阶 SGD：一阶梯度内积 + HVP 曲率项。"""
    set_seed(config.seed)
    base_train, val_dataset, train_loader, val_loader = make_pile_experiment_loaders(config)
    model = make_pile_model(config.hidden_dim)
    optimizer = optim.SGD(model.parameters(), lr=config.lr)
    loss_fn = CausalLMLoss()
    scores = torch.zeros(len(train_loader.dataset), dtype=torch.float32)
    val_imgs, val_labels = sample_validation_batch(val_loader, config.val_samples)
    snapshot, snapshot_target, scored_batches = None, snapshot_target_batch(config, train_loader), 0
    score_start = time.perf_counter(); raw_score_time = 0.0; second_order_time = 0.0

    for _ in range(config.epochs_score):
        model.train()
        for images, labels, indices in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            state_before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            was_training = enter_score_mode(model)
            _, val_grad = compute_val_grad_and_loss(model, loss_fn, val_imgs, val_labels, create_graph=True)
            step_scale = -optimizer.param_groups[0]["lr"] / images.size(0)
            step_vectors, first_terms = [], []
            raw_start = time.perf_counter()
            for i in range(images.size(0)):
                grad_i = compute_single_grad(model, loss_fn, images[i], labels[i])
                step_i = scale_dict(grad_i, step_scale)
                step_vectors.append(step_i); first_terms.append(dot_dict(val_grad, step_i))
            raw_score_time += time.perf_counter() - raw_start
            second_start = time.perf_counter()
            hvp_total = hvp_from_val_grad(val_grad, add_dicts(step_vectors), model)
            curvature_terms = [0.5 * dot_dict(step_i, hvp_total) for step_i in step_vectors]
            local_scores = combine_first_and_curvature(first_terms, curvature_terms, config)
            for i, score_i in enumerate(local_scores):
                scores[int(indices[i])] += score_i
            second_order_time += time.perf_counter() - second_start
            if snapshot is None and scored_batches >= snapshot_target:
                snapshot = LocalStepSnapshot(state_before, images.detach().cpu(), labels.detach().cpu(), indices.detach().cpu(), val_imgs.detach().cpu(), val_labels.detach().cpu(), np.asarray(local_scores), optimizer.param_groups[0]["lr"], images.size(0), "sgd")
            restore_model_mode(model, was_training)
            optimizer.zero_grad(); loss = loss_fn(model(images), labels); loss.backward(); optimizer.step(); scored_batches += 1
    score_time = time.perf_counter() - score_start
    return base_train, val_dataset, scores, snapshot, {"Score time": score_time, "Raw first-order time": raw_score_time, "HVP second-order time": second_order_time, "Second-order component time": second_order_time, "Train update time": max(0.0, score_time - raw_score_time - second_order_time)}


def compute_adam_second_order(config: PileGPT2RunConfig):
    """二阶 Adam-aware：固定 Adam 状态后的一阶项 + HVP 曲率项。"""
    set_seed(config.seed)
    base_train, val_dataset, train_loader, val_loader = make_pile_experiment_loaders(config)
    model = make_pile_model(config.hidden_dim)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, betas=(config.beta1, config.beta2), eps=config.eps, weight_decay=config.weight_decay)
    loss_fn = CausalLMLoss()
    scores = torch.zeros(len(train_loader.dataset), dtype=torch.float32)
    val_imgs, val_labels = sample_validation_batch(val_loader, config.val_samples)
    snapshot, snapshot_target, scored_batches = None, snapshot_target_batch(config, train_loader), 0
    score_start = time.perf_counter(); raw_score_time = 0.0; second_order_time = 0.0

    for _ in range(config.epochs_score):
        model.train()
        for images, labels, indices in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            state_before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            adam_state = clone_adam_state(model, optimizer)
            was_training = enter_score_mode(model)
            _, val_grad = compute_val_grad_and_loss(model, loss_fn, val_imgs, val_labels, create_graph=True)
            lr_now = optimizer.param_groups[0]["lr"]
            step_vectors, first_terms = [], []
            raw_start = time.perf_counter()
            for i in range(images.size(0)):
                grad_i = compute_single_grad(model, loss_fn, images[i], labels[i])
                update_i = adam_update_direction_from_grad(grad_i, adam_state, config.beta1, config.beta2, config.eps)
                step_i = scale_dict(update_i, -lr_now)
                step_vectors.append(step_i); first_terms.append(dot_dict(val_grad, step_i))
            raw_score_time += time.perf_counter() - raw_start
            second_start = time.perf_counter()
            hvp_total = hvp_from_val_grad(val_grad, add_dicts(step_vectors), model)
            curvature_terms = [0.5 * dot_dict(step_i, hvp_total) for step_i in step_vectors]
            local_scores = combine_first_and_curvature(first_terms, curvature_terms, config)
            for i, score_i in enumerate(local_scores):
                scores[int(indices[i])] += score_i
            second_order_time += time.perf_counter() - second_start
            if snapshot is None and scored_batches >= snapshot_target:
                snap_state = {k: {"exp_avg": v["exp_avg"].detach().cpu().clone(), "exp_avg_sq": v["exp_avg_sq"].detach().cpu().clone(), "step": int(v["step"])} for k, v in adam_state.items()}
                snapshot = LocalStepSnapshot(state_before, images.detach().cpu(), labels.detach().cpu(), indices.detach().cpu(), val_imgs.detach().cpu(), val_labels.detach().cpu(), np.asarray(local_scores), lr_now, images.size(0), "adam", adam_state=snap_state)
            restore_model_mode(model, was_training)
            optimizer.zero_grad(); loss = loss_fn(model(images), labels); loss.backward(); optimizer.step(); scored_batches += 1
    score_time = time.perf_counter() - score_start
    return base_train, val_dataset, scores, snapshot, {"Score time": score_time, "Raw first-order time": raw_score_time, "HVP second-order time": second_order_time, "Second-order component time": second_order_time, "Train update time": max(0.0, score_time - raw_score_time - second_order_time)}


def compute_adam_graph_second_order(config: PileGPT2RunConfig):
    """二阶 Graph：raw Adam-aware 一阶分数 + token 图邻域代理二阶交互。"""
    set_seed(config.seed)
    base_train, val_dataset, train_loader, val_loader = make_pile_experiment_loaders(config)
    graph_start = time.perf_counter()
    graph = build_knn_graph_from_token_blocks(base_train, config.topo_k, config.topo_feature_dim)
    graph_time = time.perf_counter() - graph_start
    model = make_pile_model(config.hidden_dim)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, betas=(config.beta1, config.beta2), eps=config.eps, weight_decay=config.weight_decay)
    loss_fn = CausalLMLoss()
    raw_scores = torch.zeros(len(train_loader.dataset), dtype=torch.float32)
    topo_scores = torch.zeros(len(train_loader.dataset), dtype=torch.float32)
    val_imgs, val_labels = sample_validation_batch(val_loader, config.val_samples)
    snapshot, snapshot_target, scored_batches = None, snapshot_target_batch(config, train_loader), 0
    score_start = time.perf_counter(); raw_score_time = 0.0; topology_time = 0.0

    for _ in range(config.epochs_score):
        model.train()
        for images, labels, indices in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            state_before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            adam_state = clone_adam_state(model, optimizer)
            was_training = enter_score_mode(model)
            _, val_grad = compute_val_grad_and_loss(model, loss_fn, val_imgs, val_labels, create_graph=False)
            lr_now = optimizer.param_groups[0]["lr"]
            raw_start = time.perf_counter(); raw_batch = []
            for i in range(images.size(0)):
                grad_i = compute_single_grad(model, loss_fn, images[i], labels[i])
                update_i = adam_update_direction_from_grad(grad_i, adam_state, config.beta1, config.beta2, config.eps)
                raw_batch.append(-lr_now * dot_dict(val_grad, update_i))
            raw_score_time += time.perf_counter() - raw_start
            batch_idx = indices.numpy(); raw_np = np.asarray(raw_batch, dtype=np.float32)
            topo_start = time.perf_counter()
            topo_np = topology_second_order_surrogate_scores(batch_idx, raw_np, raw_scores.numpy(), graph, config.topo_alpha, config.topo_lambda)
            topology_time += time.perf_counter() - topo_start
            raw_scores[batch_idx] += torch.from_numpy(raw_np)
            topo_scores[batch_idx] += torch.from_numpy(topo_np.astype(np.float32))
            if snapshot is None and scored_batches >= snapshot_target:
                snap_state = {k: {"exp_avg": v["exp_avg"].detach().cpu().clone(), "exp_avg_sq": v["exp_avg_sq"].detach().cpu().clone(), "step": int(v["step"])} for k, v in adam_state.items()}
                snapshot = LocalStepSnapshot(state_before, images.detach().cpu(), labels.detach().cpu(), indices.detach().cpu(), val_imgs.detach().cpu(), val_labels.detach().cpu(), np.asarray(raw_batch), lr_now, images.size(0), "adam", adam_state=snap_state, topo_approx_scores=np.asarray(topo_np))
            restore_model_mode(model, was_training)
            optimizer.zero_grad(); loss = loss_fn(model(images), labels); loss.backward(); optimizer.step(); scored_batches += 1
    score_time = time.perf_counter() - score_start
    timing = {"Graph build time": graph_time, "Score time": score_time, "Raw Adam score time": raw_score_time, "Raw first-order time": raw_score_time, "Topology distance time": topology_time, "Graph second-order time": topology_time, "Second-order component time": topology_time, "Train update time": max(0.0, score_time - raw_score_time - topology_time), "Total graph+score time": graph_time + score_time}
    return base_train, val_dataset, raw_scores, topo_scores, snapshot, timing


def save_and_finish_single(title: str, config: PileGPT2RunConfig, optimizer_name: str, scores, snapshot, timing, base_train, val_dataset):
    out = save_scores(config.output, scores)
    plot_out = finish_experiment(title, config, optimizer_name, base_train, val_dataset, scores, snapshot, timing)
    return out, plot_out


def save_and_finish_graph(title: str, config: PileGPT2RunConfig, graph_scores, baseline_results, graph_result, base_train, val_dataset):
    out = save_scores(config.output, graph_scores)
    plot_out = finish_graph_comparison_experiment(title, config, base_train, val_dataset, baseline_results, graph_result)
    return out, plot_out
