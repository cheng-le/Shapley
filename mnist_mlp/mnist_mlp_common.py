
"""MNIST + MLP 实验专用 common。

数据集、模型、参数默认值和图片显示都放在这里；通用 Shapley 算法来自根目录 shapley_common.py。
"""

from __future__ import annotations

import argparse
import gzip
import struct
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset

from shapley_common import *  # noqa: F401,F403
import shapley_common as core


class LocalMNIST(Dataset):
    """不依赖 torchvision 的 MNIST IDX 读取器，直接使用项目 data/MNIST/raw 下的文件。"""

    def __init__(self, root: str, train: bool):
        raw_dir = Path(root) / "MNIST" / "raw"
        image_name = "train-images-idx3-ubyte" if train else "t10k-images-idx3-ubyte"
        label_name = "train-labels-idx1-ubyte" if train else "t10k-labels-idx1-ubyte"
        self.images = self._read_images(raw_dir / image_name)
        self.targets = self._read_labels(raw_dir / label_name)

    @staticmethod
    def _open_idx(path: Path):
        if path.exists():
            return open(path, "rb")
        gz_path = path.with_suffix(path.suffix + ".gz")
        if gz_path.exists():
            return gzip.open(gz_path, "rb")
        raise FileNotFoundError(f"找不到 MNIST 文件: {path} 或 {gz_path}")

    @classmethod
    def _read_images(cls, path: Path) -> torch.Tensor:
        with cls._open_idx(path) as f:
            magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
            if magic != 2051:
                raise ValueError(f"MNIST 图像文件 magic number 不正确: {magic}")
            data = np.frombuffer(f.read(), dtype=np.uint8).copy()
        data = data.reshape(count, 1, rows, cols).astype(np.float32) / 255.0
        return torch.from_numpy(data)

    @classmethod
    def _read_labels(cls, path: Path) -> torch.Tensor:
        with cls._open_idx(path) as f:
            magic, count = struct.unpack(">II", f.read(8))
            if magic != 2049:
                raise ValueError(f"MNIST 标签文件 magic number 不正确: {magic}")
            data = np.frombuffer(f.read(), dtype=np.uint8).copy()
        if data.shape[0] != count:
            raise ValueError("MNIST 标签数量和文件头不一致。")
        return torch.from_numpy(data.astype(np.int64))

    def __len__(self) -> int:
        return int(self.targets.shape[0])

    def __getitem__(self, idx: int):
        return self.images[idx], int(self.targets[idx])


class MLP_MNIST(nn.Module):
    """MNIST 复现实验使用的简单双层 MLP。"""

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def make_model(hidden_dim: int) -> nn.Module:
    return MLP_MNIST(hidden_dim=hidden_dim).to(DEVICE)


def make_mnist_optimizer_for_eval(model: nn.Module, optimizer_name: str, config: RunConfig) -> optim.Optimizer:
    if optimizer_name == "sgd":
        return optim.SGD(model.parameters(), lr=config.lr)
    return optim.Adam(model.parameters(), lr=config.lr, betas=(config.beta1, config.beta2), eps=config.eps)


def plot_mnist_sample_grid(ax, dataset: Dataset, sample_indices: np.ndarray, title: str, scores: np.ndarray) -> None:
    ax.set_title(title, fontsize=11, pad=10)
    ax.axis("off")
    n = len(sample_indices)
    if n == 0:
        return
    cols = min(5, n)
    rows = int(np.ceil(n / cols))
    cell_w = 1.0 / cols
    cell_h = 0.92 / rows
    for pos, idx in enumerate(sample_indices):
        row = pos // cols
        col = pos % cols
        left = col * cell_w + 0.03 * cell_w
        bottom = 0.04 + (rows - row - 1) * cell_h + 0.17 * cell_h
        sub = ax.inset_axes([left, bottom, 0.94 * cell_w, 0.66 * cell_h])
        img, y = dataset[int(idx)]
        sub.imshow(img.squeeze(0).numpy(), cmap="gray")
        sub.axis("off")
        ax.text(
            left + 0.47 * cell_w,
            bottom - 0.035,
            f"i={int(idx)}  y={y}\ns={scores[int(idx)]:.2g}",
            ha="center",
            va="top",
            fontsize=7,
            transform=ax.transAxes,
        )


def install_mnist_hooks() -> None:
    core.configure_runtime(
        model_factory=make_model,
        optimizer_factory=make_mnist_optimizer_for_eval,
        loss_factory=lambda config: nn.CrossEntropyLoss(),
        sample_plotter=plot_mnist_sample_grid,
        utility_train_mode=True,
    )


def add_common_args(parser: argparse.ArgumentParser, optimizer_name: str) -> None:
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs-score", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.01 if optimizer_name == "sgd" else 1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--train-limit", type=int, default=60000, help="调试时可设小，例如 512。")
    parser.add_argument("--val-limit", type=int, default=10000, help="调试时可设小，例如 256。")
    parser.add_argument("--val-samples", type=int, default=20)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--plot-output", type=str, default="", help="保存评估总图的路径。")
    parser.add_argument("--mc-batch-size", type=int, default=12, help="局部 MC Shapley 验证使用的 batch 子集大小。")
    parser.add_argument("--mc-permutations", type=int, default=80, help="局部 MC Shapley 的随机排列数量。")
    parser.add_argument("--prune-epochs", type=int, default=2, help="删/留样本后重新训练的 epoch 数。")
    if optimizer_name in {"adam", "adam_graph"}:
        parser.add_argument("--beta1", type=float, default=0.9)
        parser.add_argument("--beta2", type=float, default=0.999)
        parser.add_argument("--eps", type=float, default=1e-8)
    if optimizer_name == "adam_graph":
        parser.add_argument("--topo-k", type=int, default=10)
        parser.add_argument("--topo-pca-dim", type=int, default=64)
        parser.add_argument("--topo-alpha", type=float, default=0.2)
        parser.add_argument("--topo-lambda", type=float, default=0.05)


def config_from_args(args: argparse.Namespace) -> RunConfig:
    return RunConfig(**vars(args))


def load_mnist_datasets(config: RunConfig) -> Tuple[Dataset, Dataset]:
    train = LocalMNIST(root="./data", train=True)
    val = LocalMNIST(root="./data", train=False)
    if config.train_limit > 0 and config.train_limit < len(train):
        train = Subset(train, list(range(config.train_limit)))
    if config.val_limit > 0 and config.val_limit < len(val):
        val = Subset(val, list(range(config.val_limit)))
    return train, val


def make_experiment_loaders(config: RunConfig) -> Tuple[Dataset, Dataset, DataLoader, DataLoader]:
    """返回训练集、验证集、带索引的训练 loader 和单样本验证 loader。"""
    install_mnist_hooks()
    base_train, val = load_mnist_datasets(config)
    train_loader = DataLoader(IndexedDataset(base_train), batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=1, shuffle=True)
    return base_train, val, train_loader, val_loader


# 导入该模块后默认注册 MNIST 钩子；CIFAR 模块会注册自己的钩子，不会使用这里的 MLP。
install_mnist_hooks()
