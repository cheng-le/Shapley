
"""CIFAR10 + ResNet18 版 In-Run Data Shapley 公共代码。

这个文件只负责 CIFAR10 数据集、ResNet18 模型和 CIFAR 专用运行参数；
通用 Shapley 公式、MC fidelity、drop/keep 和画图框架来自根目录 shapley_common.py。
"""

from __future__ import annotations

import argparse
import pickle
import tarfile
import shutil
import time
import urllib.error
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models import ResNet18_Weights, resnet18

import shapley_common as core
from shapley_common import (
    DEVICE,
    IndexedDataset,
    LocalStepSnapshot,
    add_dicts,
    adam_update_direction_from_grad,
    build_knn_graph_from_raw_pixels,
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
    topology_regularize_scores,
    topology_second_order_surrogate_scores,
)

CIFAR10_URLS = [
    "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
    "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
]
CIFAR10_ROOT = Path("data") / "CIFAR10"
CIFAR10_DIR = CIFAR10_ROOT / "cifar-10-batches-py"
CIFAR10_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
CIFAR10_STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


@dataclass
class CIFARRunConfig:
    seed: int
    batch_size: int
    epochs_score: int
    lr: float
    hidden_dim: int
    train_limit: int
    val_limit: int
    val_samples: int
    output: str
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    topo_k: int = 10
    topo_pca_dim: int = 64
    topo_alpha: float = 0.2
    topo_lambda: float = 0.05
    plot_output: str = ""
    mc_batch_size: int = 8
    mc_permutations: int = 30
    prune_epochs: int = 50
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    download: bool = False
    cifar_url: str = ""
    snapshot_after_batches: int = 0
    pretrained: bool = True
    curvature_clip_ratio: float = 0.25


def add_cifar_args(parser: argparse.ArgumentParser, optimizer_name: str) -> None:
    """CIFAR/ResNet18 实验参数；默认值偏向可调试，完整复现时可把 limit 调大。"""
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs-score", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.01 if optimizer_name == "sgd" else 1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64, help="兼容旧脚本的占位参数；torchvision ResNet18 使用标准宽度。")
    parser.add_argument("--train-limit", type=int, default=1000, help="论文 Table 5 使用 CIFAR10 的 size-1000 训练子集。")
    parser.add_argument("--val-limit", type=int, default=10000)
    parser.add_argument("--val-samples", type=int, default=20)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--plot-output", type=str, default="")
    parser.add_argument("--mc-batch-size", type=int, default=8)
    parser.add_argument("--mc-permutations", type=int, default=30)
    parser.add_argument("--prune-epochs", type=int, default=50)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--download", action="store_true", help="如果本地没有 CIFAR10，则下载 CIFAR10。")
    parser.add_argument("--cifar-url", type=str, default="", help="自定义 CIFAR10 tar.gz 下载地址；例如学校镜像或内网地址。")
    parser.add_argument("--snapshot-after-batches", type=int, default=0, help="跳过多少个训练 batch 后再保存局部 MC fidelity 快照；可用于对比训练中状态，默认使用第一个 batch。")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false", help="不加载 ImageNet 预训练 ResNet18，改用随机初始化。")
    parser.add_argument("--curvature-clip-ratio", type=float, default=0.25, help="二阶 HVP 曲率项的 trust-region 缩放比例；<0 表示不缩放，0 表示退化为一阶。")
    parser.set_defaults(pretrained=True)
    if optimizer_name in {"adam", "adam_graph"}:
        parser.add_argument("--beta1", type=float, default=0.9)
        parser.add_argument("--beta2", type=float, default=0.999)
        parser.add_argument("--eps", type=float, default=1e-8)
    if optimizer_name == "adam_graph":
        parser.add_argument("--topo-k", type=int, default=10)
        parser.add_argument("--topo-pca-dim", type=int, default=64)
        parser.add_argument("--topo-alpha", type=float, default=0.2)
        parser.add_argument("--topo-lambda", type=float, default=0.05)


def cifar_config_from_args(args: argparse.Namespace) -> CIFARRunConfig:
    return CIFARRunConfig(**vars(args))


class LocalCIFAR10(Dataset):
    """不依赖 torchvision 的 CIFAR10 读取器，直接读取官方 python pickle 数据。"""

    def __init__(self, root: str = "data", train: bool = True, download: bool = False, cifar_url: str = ""):
        self.root = Path(root) / "CIFAR10" if Path(root).name != "CIFAR10" else Path(root)
        self.cifar_url = cifar_url
        if download and not CIFAR10_DIR.exists():
            self._download()
        if not CIFAR10_DIR.exists():
            raise FileNotFoundError(
                "找不到 CIFAR10 数据。请先运行脚本并加 --download，或手动解压到 data/CIFAR10/cifar-10-batches-py。"
            )
        self.train = train
        self.data, self.targets = self._load_batches(train)

    def _download(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        archive = self.root / "cifar-10-python.tar.gz"
        if not archive.exists():
            last_error = None
            urls = [self.cifar_url] if self.cifar_url else CIFAR10_URLS
            for url in urls:
                try:
                    print(f"正在下载 CIFAR10: {url}")
                    with urllib.request.urlopen(url, timeout=30) as response, open(archive, "wb") as f:
                        shutil.copyfileobj(response, f)
                    break
                except (urllib.error.URLError, TimeoutError, OSError) as exc:
                    last_error = exc
                    if archive.exists():
                        archive.unlink()
            if not archive.exists():
                raise RuntimeError(f"CIFAR10 下载失败，请手动下载到 {archive}: {last_error}")
        print("正在解压 CIFAR10...")
        try:
            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(self.root)
        except tarfile.TarError as exc:
            raise RuntimeError(f"CIFAR10 压缩包可能损坏，请删除 {archive} 后重试。") from exc

    @staticmethod
    def _read_pickle(path: Path) -> Dict:
        # CIFAR10 官方 pickle 在新版 NumPy 下可能触发 dtype align 的弃用警告；数据本身可正常读取。
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*align should be passed.*", category=Warning)
            with open(path, "rb") as f:
                return pickle.load(f, encoding="latin1")

    def _load_batches(self, train: bool) -> Tuple[torch.Tensor, List[int]]:
        files = [f"data_batch_{i}" for i in range(1, 6)] if train else ["test_batch"]
        xs, ys = [], []
        for name in files:
            batch = self._read_pickle(CIFAR10_DIR / name)
            xs.append(batch["data"])
            ys.extend(batch["labels"])
        arr = np.concatenate(xs, axis=0).reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        data = torch.from_numpy(arr)
        return data, [int(y) for y in ys]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):
        # 默认使用 ImageNet 预训练 ResNet18，因此输入也使用 ImageNet 归一化。
        x = (self.data[idx] - IMAGENET_MEAN) / IMAGENET_STD
        return x, int(self.targets[idx])


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.relu(out)


class ResNet18CIFAR(nn.Module):
    """CIFAR 版 ResNet18：3x3 stem，无 ImageNet 的 7x7 conv/maxpool。"""

    def __init__(self, num_classes: int = 10, base_width: int = 64):
        super().__init__()
        self.in_planes = base_width
        self.conv1 = nn.Conv2d(3, base_width, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_width)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(base_width, 2, stride=1)
        self.layer2 = self._make_layer(base_width * 2, 2, stride=2)
        self.layer3 = self._make_layer(base_width * 4, 2, stride=2)
        self.layer4 = self._make_layer(base_width * 8, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_width * 8, num_classes)
        self._init_weights()

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(self.in_planes, planes, stride)]
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_planes, planes, 1))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def make_cifar_model(hidden_dim: int = 64, pretrained: bool = True) -> nn.Module:
    """默认使用 torchvision 的 ImageNet-pretrained ResNet18，并替换最后分类层为 CIFAR10。"""
    if pretrained:
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    else:
        model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model.to(DEVICE)


_ACTIVE_CONFIG: CIFARRunConfig | None = None

def install_cifar_hooks(config: CIFARRunConfig) -> None:
    """向通用 Shapley core 注册 CIFAR10 + ResNet18 的模型、优化器和图片显示逻辑。"""
    global _ACTIVE_CONFIG
    _ACTIVE_CONFIG = config
    core.configure_runtime(
        model_factory=lambda hidden_dim: make_cifar_model(hidden_dim, pretrained=_ACTIVE_CONFIG.pretrained),
        optimizer_factory=make_cifar_optimizer_for_eval,
        loss_factory=make_loss,
        sample_plotter=plot_cifar_sample_grid,
        utility_train_mode=False,
    )


def load_cifar_datasets(config: CIFARRunConfig) -> Tuple[Dataset, Dataset]:
    train = LocalCIFAR10(train=True, download=config.download, cifar_url=config.cifar_url)
    val = LocalCIFAR10(train=False, download=config.download, cifar_url=config.cifar_url)
    if config.train_limit > 0 and config.train_limit < len(train):
        train = Subset(train, list(range(config.train_limit)))
    if config.val_limit > 0 and config.val_limit < len(val):
        val = Subset(val, list(range(config.val_limit)))
    return train, val


def make_cifar_experiment_loaders(config: CIFARRunConfig) -> Tuple[Dataset, Dataset, DataLoader, DataLoader]:
    base_train, val = load_cifar_datasets(config)
    train_loader = DataLoader(IndexedDataset(base_train), batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=1, shuffle=True)
    return base_train, val, train_loader, val_loader


def make_loss(config: CIFARRunConfig) -> nn.Module:
    return nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)


def make_cifar_optimizer_for_eval(model: nn.Module, optimizer_name: str, config: CIFARRunConfig) -> optim.Optimizer:
    if optimizer_name == "sgd":
        return optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=config.weight_decay)
    return optim.Adam(model.parameters(), lr=config.lr, betas=(config.beta1, config.beta2), eps=config.eps, weight_decay=config.weight_decay)


def plot_cifar_sample_grid(ax, dataset: Dataset, sample_indices: np.ndarray, title: str, scores: np.ndarray) -> None:
    """CIFAR 图片需要反标准化并把 C,H,W 转成 H,W,C。"""
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
        img = (img.cpu() * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1)
        sub.imshow(img.permute(1, 2, 0).numpy())
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



def enter_score_mode(model: nn.Module) -> bool:
    """ResNet 有 BatchNorm；打分时冻结 BN 统计，避免 per-sample 梯度污染 running stats。"""
    was_training = model.training
    model.eval()
    return was_training


def restore_model_mode(model: nn.Module, was_training: bool) -> None:
    model.train(was_training)



def snapshot_target_batch(config: CIFARRunConfig, train_loader: DataLoader) -> int:
    """选择用于 MC fidelity 的 batch 编号；数据太少时自动退到最后一个 batch。"""
    total_batches = max(1, len(train_loader) * config.epochs_score)
    return min(max(0, config.snapshot_after_batches), total_batches - 1)


def compute_sgd_first_order(config: CIFARRunConfig):
    """一阶 SGD：phi_i = -(lr / B) * <g_val, g_i>。"""
    install_cifar_hooks(config)
    set_seed(config.seed)
    base_train, val_dataset, train_loader, val_loader = make_cifar_experiment_loaders(config)
    model = make_cifar_model(config.hidden_dim, pretrained=config.pretrained)
    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=config.weight_decay)
    loss_fn = make_loss(config)
    scores = torch.zeros(len(train_loader.dataset), dtype=torch.float32)
    val_imgs, val_labels = sample_validation_batch(val_loader, config.val_samples)
    snapshot = None
    snapshot_target = snapshot_target_batch(config, train_loader)
    scored_batches = 0
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
                snapshot = LocalStepSnapshot(state_before, images.detach().cpu().clone(), labels.detach().cpu().clone(), indices.detach().cpu().clone(), val_imgs.detach().cpu().clone(), val_labels.detach().cpu().clone(), np.asarray(local_scores, dtype=np.float64), optimizer.param_groups[0]["lr"], images.size(0), "sgd")
            restore_model_mode(model, was_training)
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            scored_batches += 1
    return base_train, val_dataset, scores, snapshot, {"Score time": time.perf_counter() - score_start}


def compute_adam_first_order(config: CIFARRunConfig):
    """Adam-aware 一阶：固定 Adam 状态后，用单样本梯度进入 Adam 更新式。"""
    install_cifar_hooks(config)
    set_seed(config.seed)
    base_train, val_dataset, train_loader, val_loader = make_cifar_experiment_loaders(config)
    model = make_cifar_model(config.hidden_dim, pretrained=config.pretrained)
    optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=(config.beta1, config.beta2), eps=config.eps, weight_decay=config.weight_decay)
    loss_fn = make_loss(config)
    scores = torch.zeros(len(train_loader.dataset), dtype=torch.float32)
    val_imgs, val_labels = sample_validation_batch(val_loader, config.val_samples)
    snapshot = None
    snapshot_target = snapshot_target_batch(config, train_loader)
    scored_batches = 0
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
                snapshot = LocalStepSnapshot(state_before, images.detach().cpu().clone(), labels.detach().cpu().clone(), indices.detach().cpu().clone(), val_imgs.detach().cpu().clone(), val_labels.detach().cpu().clone(), np.asarray(local_scores, dtype=np.float64), lr_now, images.size(0), "adam", adam_state=snap_state)
            restore_model_mode(model, was_training)
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            scored_batches += 1
    return base_train, val_dataset, scores, snapshot, {"Score time": time.perf_counter() - score_start}


def compute_adam_graph_first_order(config: CIFARRunConfig):
    """Adam-aware 一阶 + 图扩散：先算 raw Adam 分数，再用 KNN 图做邻域稳定化。"""
    install_cifar_hooks(config)
    set_seed(config.seed)
    base_train, val_dataset, train_loader, val_loader = make_cifar_experiment_loaders(config)
    graph_start = time.perf_counter()
    graph = build_knn_graph_from_raw_pixels(base_train, config.topo_k, config.topo_pca_dim, config.seed)
    graph_time = time.perf_counter() - graph_start
    model = make_cifar_model(config.hidden_dim, pretrained=config.pretrained)
    optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=(config.beta1, config.beta2), eps=config.eps, weight_decay=config.weight_decay)
    loss_fn = make_loss(config)
    raw_scores = torch.zeros(len(train_loader.dataset), dtype=torch.float32)
    topo_scores = torch.zeros(len(train_loader.dataset), dtype=torch.float32)
    val_imgs, val_labels = sample_validation_batch(val_loader, config.val_samples)
    snapshot = None
    snapshot_target = snapshot_target_batch(config, train_loader)
    scored_batches = 0
    score_start = time.perf_counter()
    raw_score_time = 0.0
    topology_time = 0.0

    for _ in range(config.epochs_score):
        model.train()
        for images, labels, indices in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            state_before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            adam_state = clone_adam_state(model, optimizer)
            was_training = enter_score_mode(model)
            _, val_grad = compute_val_grad_and_loss(model, loss_fn, val_imgs, val_labels, create_graph=False)
            lr_now = optimizer.param_groups[0]["lr"]
            raw_start = time.perf_counter()
            raw_batch = []
            for i in range(images.size(0)):
                grad_i = compute_single_grad(model, loss_fn, images[i], labels[i])
                update_i = adam_update_direction_from_grad(grad_i, adam_state, config.beta1, config.beta2, config.eps)
                raw_batch.append(-lr_now * dot_dict(val_grad, update_i))
            raw_score_time += time.perf_counter() - raw_start
            batch_idx = indices.numpy()
            raw_np = np.asarray(raw_batch, dtype=np.float32)
            topo_start = time.perf_counter()
            topo_np = topology_regularize_scores(batch_idx, raw_np, raw_scores.numpy(), graph, config.topo_alpha, config.topo_lambda)
            topology_time += time.perf_counter() - topo_start
            raw_scores[batch_idx] += torch.from_numpy(raw_np)
            topo_scores[batch_idx] += torch.from_numpy(topo_np.astype(np.float32))
            if snapshot is None and scored_batches >= snapshot_target:
                snap_state = {k: {"exp_avg": v["exp_avg"].detach().cpu().clone(), "exp_avg_sq": v["exp_avg_sq"].detach().cpu().clone(), "step": int(v["step"])} for k, v in adam_state.items()}
                snapshot = LocalStepSnapshot(state_before, images.detach().cpu().clone(), labels.detach().cpu().clone(), indices.detach().cpu().clone(), val_imgs.detach().cpu().clone(), val_labels.detach().cpu().clone(), np.asarray(raw_batch, dtype=np.float64), lr_now, images.size(0), "adam", adam_state=snap_state, topo_approx_scores=np.asarray(topo_np, dtype=np.float64))
            restore_model_mode(model, was_training)
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            scored_batches += 1
    score_time = time.perf_counter() - score_start
    timing = {"Graph build time": graph_time, "Score time": score_time, "Raw Adam score time": raw_score_time, "Raw first-order time": raw_score_time, "Topology distance time": topology_time, "Train update time": max(0.0, score_time - raw_score_time - topology_time), "Total graph+score time": graph_time + score_time}
    return base_train, val_dataset, raw_scores, topo_scores, snapshot, timing



def combine_first_and_curvature(first_terms: List[float], curvature_terms: List[float], config: CIFARRunConfig) -> List[float]:
    """组合一阶项和二阶曲率项；预训练 ResNet18 上对 HVP 做信任域缩放以避免曲率噪声压过一阶信号。"""
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


def compute_sgd_second_order(config: CIFARRunConfig):
    """二阶 SGD：一阶梯度内积 + HVP 曲率项。"""
    install_cifar_hooks(config)
    set_seed(config.seed)
    base_train, val_dataset, train_loader, val_loader = make_cifar_experiment_loaders(config)
    model = make_cifar_model(config.hidden_dim, pretrained=config.pretrained)
    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=config.weight_decay)
    loss_fn = make_loss(config)
    scores = torch.zeros(len(train_loader.dataset), dtype=torch.float32)
    val_imgs, val_labels = sample_validation_batch(val_loader, config.val_samples)
    snapshot = None
    snapshot_target = snapshot_target_batch(config, train_loader)
    scored_batches = 0
    score_start = time.perf_counter()
    raw_score_time = 0.0
    second_order_time = 0.0

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
                step_vectors.append(step_i)
                first_terms.append(dot_dict(val_grad, step_i))
            raw_score_time += time.perf_counter() - raw_start
            second_start = time.perf_counter()
            hvp_total = hvp_from_val_grad(val_grad, add_dicts(step_vectors), model)
            curvature_terms = [0.5 * dot_dict(step_i, hvp_total) for step_i in step_vectors]
            local_scores = combine_first_and_curvature(first_terms, curvature_terms, config)
            for i, score_i in enumerate(local_scores):
                scores[int(indices[i])] += score_i
            second_order_time += time.perf_counter() - second_start
            if snapshot is None and scored_batches >= snapshot_target:
                snapshot = LocalStepSnapshot(state_before, images.detach().cpu().clone(), labels.detach().cpu().clone(), indices.detach().cpu().clone(), val_imgs.detach().cpu().clone(), val_labels.detach().cpu().clone(), np.asarray(local_scores, dtype=np.float64), optimizer.param_groups[0]["lr"], images.size(0), "sgd")
            restore_model_mode(model, was_training)
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            scored_batches += 1
    score_time = time.perf_counter() - score_start
    timing = {"Score time": score_time, "Raw first-order time": raw_score_time, "HVP second-order time": second_order_time, "Second-order component time": second_order_time, "Train update time": max(0.0, score_time - raw_score_time - second_order_time)}
    return base_train, val_dataset, scores, snapshot, timing


def compute_adam_second_order(config: CIFARRunConfig):
    """二阶 Adam-aware：固定 Adam 状态后的一阶项 + HVP 曲率项。"""
    install_cifar_hooks(config)
    set_seed(config.seed)
    base_train, val_dataset, train_loader, val_loader = make_cifar_experiment_loaders(config)
    model = make_cifar_model(config.hidden_dim, pretrained=config.pretrained)
    optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=(config.beta1, config.beta2), eps=config.eps, weight_decay=config.weight_decay)
    loss_fn = make_loss(config)
    scores = torch.zeros(len(train_loader.dataset), dtype=torch.float32)
    val_imgs, val_labels = sample_validation_batch(val_loader, config.val_samples)
    snapshot = None
    snapshot_target = snapshot_target_batch(config, train_loader)
    scored_batches = 0
    score_start = time.perf_counter()
    raw_score_time = 0.0
    second_order_time = 0.0

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
                step_vectors.append(step_i)
                first_terms.append(dot_dict(val_grad, step_i))
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
                snapshot = LocalStepSnapshot(state_before, images.detach().cpu().clone(), labels.detach().cpu().clone(), indices.detach().cpu().clone(), val_imgs.detach().cpu().clone(), val_labels.detach().cpu().clone(), np.asarray(local_scores, dtype=np.float64), lr_now, images.size(0), "adam", adam_state=snap_state)
            restore_model_mode(model, was_training)
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            scored_batches += 1
    score_time = time.perf_counter() - score_start
    timing = {"Score time": score_time, "Raw first-order time": raw_score_time, "HVP second-order time": second_order_time, "Second-order component time": second_order_time, "Train update time": max(0.0, score_time - raw_score_time - second_order_time)}
    return base_train, val_dataset, scores, snapshot, timing


def compute_adam_graph_second_order(config: CIFARRunConfig):
    """二阶 Graph：raw Adam-aware 一阶分数 + 图邻域距离代理二阶交互。"""
    install_cifar_hooks(config)
    set_seed(config.seed)
    base_train, val_dataset, train_loader, val_loader = make_cifar_experiment_loaders(config)
    graph_start = time.perf_counter()
    graph = build_knn_graph_from_raw_pixels(base_train, config.topo_k, config.topo_pca_dim, config.seed)
    graph_time = time.perf_counter() - graph_start
    model = make_cifar_model(config.hidden_dim, pretrained=config.pretrained)
    optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=(config.beta1, config.beta2), eps=config.eps, weight_decay=config.weight_decay)
    loss_fn = make_loss(config)
    raw_scores = torch.zeros(len(train_loader.dataset), dtype=torch.float32)
    topo_scores = torch.zeros(len(train_loader.dataset), dtype=torch.float32)
    val_imgs, val_labels = sample_validation_batch(val_loader, config.val_samples)
    snapshot = None
    snapshot_target = snapshot_target_batch(config, train_loader)
    scored_batches = 0
    score_start = time.perf_counter()
    raw_score_time = 0.0
    topology_time = 0.0

    for _ in range(config.epochs_score):
        model.train()
        for images, labels, indices in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            state_before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            adam_state = clone_adam_state(model, optimizer)
            was_training = enter_score_mode(model)
            _, val_grad = compute_val_grad_and_loss(model, loss_fn, val_imgs, val_labels, create_graph=False)
            lr_now = optimizer.param_groups[0]["lr"]
            raw_start = time.perf_counter()
            raw_batch = []
            for i in range(images.size(0)):
                grad_i = compute_single_grad(model, loss_fn, images[i], labels[i])
                update_i = adam_update_direction_from_grad(grad_i, adam_state, config.beta1, config.beta2, config.eps)
                raw_batch.append(-lr_now * dot_dict(val_grad, update_i))
            raw_score_time += time.perf_counter() - raw_start
            batch_idx = indices.numpy()
            raw_np = np.asarray(raw_batch, dtype=np.float32)
            topo_start = time.perf_counter()
            topo_np = topology_second_order_surrogate_scores(batch_idx, raw_np, raw_scores.numpy(), graph, config.topo_alpha, config.topo_lambda)
            topology_time += time.perf_counter() - topo_start
            raw_scores[batch_idx] += torch.from_numpy(raw_np)
            topo_scores[batch_idx] += torch.from_numpy(topo_np.astype(np.float32))
            if snapshot is None and scored_batches >= snapshot_target:
                snap_state = {k: {"exp_avg": v["exp_avg"].detach().cpu().clone(), "exp_avg_sq": v["exp_avg_sq"].detach().cpu().clone(), "step": int(v["step"])} for k, v in adam_state.items()}
                snapshot = LocalStepSnapshot(state_before, images.detach().cpu().clone(), labels.detach().cpu().clone(), indices.detach().cpu().clone(), val_imgs.detach().cpu().clone(), val_labels.detach().cpu().clone(), np.asarray(raw_batch, dtype=np.float64), lr_now, images.size(0), "adam", adam_state=snap_state, topo_approx_scores=np.asarray(topo_np, dtype=np.float64))
            restore_model_mode(model, was_training)
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            scored_batches += 1
    score_time = time.perf_counter() - score_start
    timing = {"Graph build time": graph_time, "Score time": score_time, "Raw Adam score time": raw_score_time, "Raw first-order time": raw_score_time, "Topology distance time": topology_time, "Graph second-order time": topology_time, "Second-order component time": topology_time, "Train update time": max(0.0, score_time - raw_score_time - topology_time), "Total graph+score time": graph_time + score_time}
    return base_train, val_dataset, raw_scores, topo_scores, snapshot, timing


def save_and_finish_single(title: str, config: CIFARRunConfig, optimizer_name: str, scores, snapshot, timing, base_train, val_dataset):
    out = save_scores(config.output, scores)
    plot_out = finish_experiment(title, config, optimizer_name, base_train, val_dataset, scores, snapshot, timing)
    return out, plot_out
