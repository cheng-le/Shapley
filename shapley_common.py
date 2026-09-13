"""数据集/模型无关的 In-Run Data Shapley 通用工具。

MNIST+MLP 和 CIFAR10+ResNet18 各自在自己的 common 文件里注册模型、优化器、loss 和样本可视化；
这里仅保留梯度、HVP、Adam-aware、Graph、MC fidelity、drop/keep 和画图框架等通用逻辑。
"""

import random
from dataclasses import dataclass
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.stats import pearsonr, spearmanr
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Dataset, Subset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    """固定随机种子，保证同一份数据顺序下分数可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class IndexedDataset(Dataset):
    """给基础数据集额外返回全局样本编号，用来累积分数。"""

    def __init__(self, base_dataset: Dataset):
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        x, y = self.base_dataset[idx]
        return x, y, idx


_MODEL_FACTORY = None
_OPTIMIZER_FACTORY = None
_LOSS_FACTORY = None
_PLOT_SAMPLE_GRID = None
_EVALUATOR = None
_UTILITY_TRAIN_MODE = True


def configure_runtime(model_factory, optimizer_factory=None, loss_factory=None, sample_plotter=None, utility_train_mode=True, evaluator=None) -> None:
    """注册当前实验的数据集/模型相关钩子；core 本身不绑定 MNIST、CIFAR 或具体模型。"""
    global _MODEL_FACTORY, _OPTIMIZER_FACTORY, _LOSS_FACTORY, _PLOT_SAMPLE_GRID, _UTILITY_TRAIN_MODE, _EVALUATOR
    _MODEL_FACTORY = model_factory
    _OPTIMIZER_FACTORY = optimizer_factory
    _LOSS_FACTORY = loss_factory
    _PLOT_SAMPLE_GRID = sample_plotter
    _UTILITY_TRAIN_MODE = utility_train_mode
    _EVALUATOR = evaluator


def make_model(hidden_dim: int) -> nn.Module:
    if _MODEL_FACTORY is None:
        raise RuntimeError("尚未注册模型工厂，请先调用 configure_runtime(...)")
    return _MODEL_FACTORY(hidden_dim)


def make_eval_loss(config) -> nn.Module:
    if _LOSS_FACTORY is not None:
        return _LOSS_FACTORY(config)
    return nn.CrossEntropyLoss()


def zero_model_grads(model: nn.Module) -> None:
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()


def named_params(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: p for name, p in model.named_parameters() if p.requires_grad}


def flatten_tensor_dict(values: Dict[str, torch.Tensor], param_order: Iterable[str]) -> torch.Tensor:
    """按固定参数顺序拼接向量，避免 dot-product 时参数错位。"""
    return torch.cat([values[name].reshape(-1) for name in param_order])


def grad_dict_from_loss(
    model: nn.Module,
    loss: torch.Tensor,
    create_graph: bool = False,
) -> Dict[str, torch.Tensor]:
    params = named_params(model)
    grads = torch.autograd.grad(
        loss,
        tuple(params.values()),
        create_graph=create_graph,
        retain_graph=create_graph,
        allow_unused=False,
    )
    return {name: grad for name, grad in zip(params.keys(), grads)}


def compute_val_grad_and_loss(
    model: nn.Module,
    loss_fn: nn.Module,
    val_imgs: torch.Tensor,
    val_labels: torch.Tensor,
    create_graph: bool,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """计算验证损失和验证梯度；二阶算法需要保留计算图来做 HVP。"""
    logits_val = model(val_imgs)
    val_loss = loss_fn(logits_val, val_labels)
    grad_val = grad_dict_from_loss(model, val_loss, create_graph=create_graph)
    return val_loss, grad_val


def compute_single_grad(
    model: nn.Module,
    loss_fn: nn.Module,
    image: torch.Tensor,
    label: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """朴素 per-sample 梯度；小规模实验可直接使用，便于先保证公式正确。"""
    logits = model(image.unsqueeze(0))
    loss = loss_fn(logits, label.unsqueeze(0))
    return grad_dict_from_loss(model, loss, create_graph=False)


def hvp_from_val_grad(
    val_grad: Dict[str, torch.Tensor],
    vector: Dict[str, torch.Tensor],
    model: nn.Module,
) -> Dict[str, torch.Tensor]:
    """计算验证 Hessian 乘向量 H_val v，不显式构造 Hessian 矩阵。"""
    params = named_params(model)
    dot = sum((val_grad[name] * vector[name]).sum() for name in params.keys())
    hvp = torch.autograd.grad(dot, tuple(params.values()), retain_graph=True, allow_unused=False)
    return {name: h.detach() for name, h in zip(params.keys(), hvp)}


def dot_dict(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> float:
    return float(sum((a[name].detach() * b[name].detach()).sum().item() for name in a.keys()))


def scale_dict(a: Dict[str, torch.Tensor], scale: float) -> Dict[str, torch.Tensor]:
    return {name: value.detach() * scale for name, value in a.items()}


def add_dicts(items: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    names = items[0].keys()
    return {name: sum(item[name] for item in items).detach() for name in names}


def safe_step_int(step_obj) -> int:
    if isinstance(step_obj, torch.Tensor):
        return int(step_obj.item())
    return int(step_obj)


def clone_adam_state(model: nn.Module, optimizer: optim.Optimizer) -> Dict[str, Dict[str, torch.Tensor | int]]:
    """复制 Adam 当前状态；每一步打分必须使用 optimizer.step() 之前的状态。"""
    state_snapshot = {}
    for name, p in model.named_parameters():
        state = optimizer.state[p]
        if len(state) == 0:
            exp_avg = torch.zeros_like(p, device=p.device)
            exp_avg_sq = torch.zeros_like(p, device=p.device)
            step = 0
        else:
            exp_avg = state["exp_avg"].detach().clone()
            exp_avg_sq = state["exp_avg_sq"].detach().clone()
            step = safe_step_int(state["step"])
        state_snapshot[name] = {"exp_avg": exp_avg, "exp_avg_sq": exp_avg_sq, "step": step}
    return state_snapshot


def adam_update_direction_from_grad(
    sample_grad: Dict[str, torch.Tensor],
    adam_state: Dict[str, Dict[str, torch.Tensor | int]],
    beta1: float,
    beta2: float,
    eps: float,
) -> Dict[str, torch.Tensor]:
    """论文中的 Adam-aware 固定状态近似：把单样本梯度放入 Adam 更新式。"""
    update_dir = {}
    for name, g in sample_grad.items():
        prev_m = adam_state[name]["exp_avg"].to(g.device)
        prev_v = adam_state[name]["exp_avg_sq"].to(g.device)
        prev_step = int(adam_state[name]["step"])
        step_t = prev_step + 1
        m_t = beta1 * prev_m + (1.0 - beta1) * g
        v_t = beta2 * prev_v + (1.0 - beta2) * (g * g)
        m_hat = m_t / (1.0 - beta1 ** step_t)
        v_hat = v_t / (1.0 - beta2 ** step_t)
        update_dir[name] = (m_hat / (torch.sqrt(v_hat) + eps)).detach()
    return update_dir


@dataclass
class TopologyGraph:
    """固定 K 的有向 KNN 图；density 用来惩罚全局冗余/过密样本。"""

    neighbors: np.ndarray
    weights: np.ndarray
    density: np.ndarray


def build_knn_graph_from_raw_pixels(base_dataset: Dataset, k: int, pca_dim: int, seed: int) -> TopologyGraph:
    """用样本张量展平后的特征构建离线 KNN 图，作为图正则/扩散的邻接矩阵。"""
    xs = []
    for i in range(len(base_dataset)):
        xs.append(base_dataset[i][0].view(-1).numpy())
    x_mat = np.stack(xs).astype(np.float32)
    x_mat /= np.linalg.norm(x_mat, axis=1, keepdims=True) + 1e-12
    if pca_dim > 0 and pca_dim < x_mat.shape[1]:
        x_mat = PCA(n_components=pca_dim, random_state=seed).fit_transform(x_mat).astype(np.float32)
        x_mat /= np.linalg.norm(x_mat, axis=1, keepdims=True) + 1e-12

    nbrs = NearestNeighbors(n_neighbors=min(k + 1, len(base_dataset)), metric="cosine", n_jobs=-1)
    nbrs.fit(x_mat)
    dist, ind = nbrs.kneighbors(x_mat)
    neighbors = ind[:, 1:].astype(np.int64, copy=True)
    raw_weights = np.clip(1.0 - dist[:, 1:], 0.0, None).astype(np.float32, copy=True)
    density = raw_weights.mean(axis=1).astype(np.float32)
    row_sum = raw_weights.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0.0] = 1.0
    weights = raw_weights / row_sum
    return TopologyGraph(neighbors=neighbors, weights=weights, density=density)


def laplacian_smooth_scores(
    scores: np.ndarray,
    graph: TopologyGraph,
    regularization: float,
) -> np.ndarray:
    """Smooth a complete score vector with a symmetric graph Laplacian.

    Solving ``(I + lambda L) x = scores`` preserves the constant component and
    contracts graph-frequency noise. Unlike the online graph helper below, this
    operation compares complete per-example scores at one consistent scale.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or len(scores) != len(graph.neighbors):
        raise ValueError("scores and graph must contain the same number of examples")
    if regularization < 0.0:
        raise ValueError("regularization must be non-negative")
    if regularization == 0.0 or len(scores) == 0:
        return scores.copy()

    rows = np.repeat(np.arange(len(scores), dtype=np.int64), graph.neighbors.shape[1])
    adjacency = sp.csr_matrix(
        (graph.weights.reshape(-1), (rows, graph.neighbors.reshape(-1))),
        shape=(len(scores), len(scores)),
        dtype=np.float64,
    )
    adjacency = 0.5 * (adjacency + adjacency.T)
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    laplacian = sp.diags(degree, format="csr") - adjacency
    system = sp.eye(len(scores), format="csr") + regularization * laplacian
    return np.asarray(sp.linalg.spsolve(system, scores), dtype=np.float64)


def conservative_graph_scores(
    scores: np.ndarray,
    labels: np.ndarray,
    graph: TopologyGraph,
    smoothing: float = 0.2,
    consistency_penalty: float = 1.0,
) -> np.ndarray:
    """Use graph support without allowing low-value nodes to inherit value.

    Only observed-label-compatible edges provide support. A point can be
    pulled down when its score lacks neighbor support, but it is never raised
    above its raw score. Label-inconsistent neighborhoods receive an additional
    scale-aware penalty, which is useful for corrupted-label selection tasks.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    if scores.ndim != 1 or labels.shape != scores.shape or len(scores) != len(graph.neighbors):
        raise ValueError("scores, labels, and graph must contain the same number of examples")
    if not 0.0 <= smoothing <= 1.0 or consistency_penalty < 0.0:
        raise ValueError("invalid graph regularization strength")

    compatible = labels[graph.neighbors] == labels[:, None]
    compatible_weights = graph.weights.astype(np.float64) * compatible
    agreement = np.clip(compatible_weights.sum(axis=1), 0.0, 1.0)
    support = (compatible_weights * scores[graph.neighbors]).sum(axis=1)
    support = np.divide(support, agreement, out=scores.copy(), where=agreement > 1e-12)
    unsupported_excess = np.maximum(scores - support, 0.0)
    scale = float(np.std(scores))
    return scores - smoothing * unsupported_excess - consistency_penalty * scale * (1.0 - agreement)


def _match_raw_scale(term: np.ndarray, raw_batch_scores: np.ndarray) -> np.ndarray:
    """把图项缩放到 raw Shapley 分数的尺度，避免图结构压过真实贡献。"""
    term = np.asarray(term, dtype=np.float32)
    raw_scale = float(np.std(raw_batch_scores))
    term_scale = float(np.std(term))
    if raw_scale < 1e-12 or term_scale < 1e-12:
        return np.zeros_like(term, dtype=np.float32)
    return ((term - float(np.mean(term))) / (term_scale + 1e-12) * raw_scale).astype(np.float32)


def _batch_graph_terms(
    batch_indices: np.ndarray,
    raw_batch_scores: np.ndarray,
    raw_score_buffer: np.ndarray,
    graph: TopologyGraph,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """严格 O(batch_size * topo_k) 的图项计算。"""
    batch_indices = np.asarray(batch_indices, dtype=np.int64)
    raw_batch_scores = np.asarray(raw_batch_scores, dtype=np.float32)
    neigh = graph.neighbors[batch_indices]
    weights = graph.weights[batch_indices]
    diffusion = (weights * raw_score_buffer[neigh]).sum(axis=1).astype(np.float32)
    pair_signal = np.zeros(len(batch_indices), dtype=np.float32)
    crowding = graph.density[batch_indices].astype(np.float32, copy=True)

    # pair_signal 只需要检查固定 K 个邻居是否也在当前 batch 中。
    local_pos = {int(global_idx): pos for pos, global_idx in enumerate(batch_indices)}
    for row_pos in range(len(batch_indices)):
        for nbr, weight in zip(neigh[row_pos], weights[row_pos]):
            nbr_pos = local_pos.get(int(nbr))
            if nbr_pos is None:
                continue
            signal = float(weight) * float(raw_batch_scores[nbr_pos])
            diffusion[row_pos] += signal
            pair_signal[row_pos] += signal
    return diffusion, pair_signal, crowding


def topology_regularize_scores(
    batch_indices: np.ndarray,
    raw_batch_scores: np.ndarray,
    raw_score_buffer: np.ndarray,
    graph: TopologyGraph,
    alpha: float,
    lambda_reg: float,
) -> np.ndarray:
    """图增强：只做邻域价值扩散；不把密度直接当作有害性加入 Shapley 分数。"""
    diffusion, _, _ = _batch_graph_terms(batch_indices, raw_batch_scores, raw_score_buffer, graph)
    diffusion_term = _match_raw_scale(diffusion, raw_batch_scores)
    return raw_batch_scores + alpha * diffusion_term



def topology_second_order_surrogate_scores(
    batch_indices: np.ndarray,
    raw_batch_scores: np.ndarray,
    raw_score_buffer: np.ndarray,
    graph: TopologyGraph,
    alpha: float,
    lambda_reg: float,
) -> np.ndarray:
    """
    图代理二阶项：不用 HVP/显式 Hessian，而用样本图上的邻域距离近似二阶交互。

    raw_batch_scores 是 Adam-aware 一阶贡献；diffusion 表示全局邻域价值传播；
    pair_signal 只看当前 batch 内有边相连样本的相互作用，作为 d_i^T H d_j 的图代理；
    图项只平滑/近似交互，不把全局密度直接当作负价值；密度更适合作为多样性选择约束。
    """
    diffusion, pair_signal, _ = _batch_graph_terms(batch_indices, raw_batch_scores, raw_score_buffer, graph)
    diffusion_term = _match_raw_scale(diffusion, raw_batch_scores)
    pair_term = _match_raw_scale(pair_signal, raw_batch_scores)
    return raw_batch_scores + alpha * diffusion_term + 0.5 * alpha * pair_term


@dataclass
class RunConfig:
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
    mc_batch_size: int = 12
    mc_permutations: int = 80
    prune_epochs: int = 2


def sample_validation_batch(val_loader: DataLoader, count: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """从验证集按类别均衡采样 reference，避免 Shapley 分数被少数数字类别主导。"""
    dataset = val_loader.dataset
    labels = get_dataset_labels(dataset)
    rng = np.random.default_rng(SEED if "SEED" in globals() else 0)
    xs, ys = [], []
    classes = np.unique(labels)
    per_class = max(1, int(np.ceil(count / len(classes))))
    for c in classes:
        idx_c = np.where(labels == c)[0]
        if len(idx_c) == 0:
            continue
        chosen = rng.choice(idx_c, size=min(per_class, len(idx_c)), replace=False)
        for idx in chosen:
            x, y = dataset[int(idx)]
            xs.append(x.unsqueeze(0))
            if isinstance(y, torch.Tensor):
                ys.append(y.unsqueeze(0))
            else:
                ys.append(torch.tensor([int(y)], dtype=torch.long))
            if len(xs) >= count:
                return torch.cat(xs, dim=0).to(DEVICE), torch.cat(ys, dim=0).to(DEVICE)
    return torch.cat(xs, dim=0).to(DEVICE), torch.cat(ys, dim=0).to(DEVICE)


def save_scores(path: str, scores: torch.Tensor, extra: Optional[Dict[str, torch.Tensor]] = None) -> Path:
    """保存分数，默认用 torch 格式，便于后续继续做 valuation/剪枝实验。"""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"scores": scores.cpu()}
    if extra:
        payload.update({k: v.cpu() for k, v in extra.items()})
    torch.save(payload, out)
    return out


def summarize_scores(name: str, scores: torch.Tensor, topk: int = 5) -> None:
    scores_cpu = scores.detach().cpu()
    print(f"{name}: mean={scores_cpu.mean().item():.6g}, std={scores_cpu.std().item():.6g}")
    helpful = torch.argsort(scores_cpu)[:topk]
    harmful = torch.argsort(scores_cpu, descending=True)[:topk]
    print("最有帮助(分数最低):", [(int(i), float(scores_cpu[i])) for i in helpful])
    print("最可能有害(分数最高):", [(int(i), float(scores_cpu[i])) for i in harmful])


@dataclass
class LocalStepSnapshot:
    """保存一个训练 step，用于局部 MC Shapley 验证近似公式是否靠谱。"""

    state_dict: Dict[str, torch.Tensor]
    images: torch.Tensor
    labels: torch.Tensor
    indices: torch.Tensor
    val_imgs: torch.Tensor
    val_labels: torch.Tensor
    approx_scores: np.ndarray
    lr: float
    batch_size_ref: int
    optimizer_name: str
    adam_state: Optional[Dict[str, Dict[str, torch.Tensor | int]]] = None
    topo_approx_scores: Optional[np.ndarray] = None


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, loss_fn: nn.Module) -> Tuple[float, float]:
    if _EVALUATOR is not None:
        return _EVALUATOR(model, loader, loss_fn)
    model.eval()
    total_loss, total_correct, total_count = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss = loss_fn(logits, y)
        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_count += x.size(0)
    return total_loss / max(total_count, 1), total_correct / max(total_count, 1)


def make_optimizer_for_eval(model: nn.Module, optimizer_name: str, config: RunConfig) -> optim.Optimizer:
    if _OPTIMIZER_FACTORY is not None:
        return _OPTIMIZER_FACTORY(model, optimizer_name, config)
    if optimizer_name == "sgd":
        return optim.SGD(model.parameters(), lr=config.lr)
    return optim.Adam(model.parameters(), lr=config.lr, betas=(config.beta1, config.beta2), eps=config.eps)


def train_one_epoch_eval(model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer, loss_fn: nn.Module) -> None:
    model.train()
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()


def get_dataset_labels(dataset: Dataset) -> np.ndarray:
    """兼容普通 Dataset 和 Subset，返回当前训练集合的标签数组。"""
    if isinstance(dataset, Subset):
        base_labels = get_dataset_labels(dataset.dataset)
        return base_labels[np.asarray(dataset.indices, dtype=int)]
    if hasattr(dataset, "targets"):
        targets = dataset.targets
        if isinstance(targets, torch.Tensor):
            return targets.cpu().numpy().astype(int)
        return np.asarray(targets, dtype=int)
    return np.asarray([int(dataset[i][1]) for i in range(len(dataset))], dtype=int)


def class_balanced_pick(labels: np.ndarray, scores: np.ndarray, ratio: float, mode: str, seed: int) -> np.ndarray:
    """按类别等比例选择样本，避免只删/留某几个数字类别。"""
    rng = np.random.default_rng(seed)
    picked = []
    for c in np.unique(labels):
        idx_c = np.where(labels == c)[0]
        n_pick = max(1, int(round(len(idx_c) * ratio)))
        if mode == "helpful":
            chosen = idx_c[np.argsort(scores[idx_c])[:n_pick]]
        elif mode == "harmful":
            chosen = idx_c[np.argsort(scores[idx_c])[-n_pick:]]
        elif mode == "random":
            chosen = rng.choice(idx_c, size=n_pick, replace=False)
        else:
            raise ValueError(mode)
        picked.append(chosen)
    return np.concatenate(picked) if picked else np.array([], dtype=int)


def selected_score_means(labels: np.ndarray, scores: np.ndarray, ratio: float, seed: int) -> Tuple[float, float]:
    """返回 class-balanced helpful/harmful 选择集合的平均分，用来检查符号方向。"""
    helpful_idx = class_balanced_pick(labels, scores, ratio, "helpful", seed)
    harmful_idx = class_balanced_pick(labels, scores, ratio, "harmful", seed)
    return float(scores[helpful_idx].mean()), float(scores[harmful_idx].mean())


def run_retrain_curve(train_dataset: Dataset, val_dataset: Dataset, optimizer_name: str, config: RunConfig) -> float:
    # 每个 subset 重训都使用同一初始化和 shuffle 随机源，使曲线差异主要来自数据选择。
    set_seed(config.seed)
    # ResNet 的 BatchNorm 在最后一个 batch 只有 1 个样本时会报错；只丢弃这个非法尾批。
    drop_last = (not _UTILITY_TRAIN_MODE) and len(train_dataset) > config.batch_size and len(train_dataset) % config.batch_size == 1
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=drop_last)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    model = make_model(config.hidden_dim)
    optimizer = make_optimizer_for_eval(model, optimizer_name, config)
    loss_fn = make_eval_loss(config)
    for _ in range(config.prune_epochs):
        train_one_epoch_eval(model, train_loader, optimizer, loss_fn)
    _, acc = evaluate_model(model, val_loader, loss_fn)
    return acc


def run_drop_keep_experiments(
    base_train: Dataset,
    val_dataset: Dataset,
    scores: torch.Tensor,
    optimizer_name: str,
    config: RunConfig,
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]], List[float], List[float]]:
    """复现数据估值常用 sanity check：删好/坏样本、只保留好/坏样本后重新训练。"""
    drop_ratios = [0.05, 0.10, 0.20]
    keep_ratios = [0.05, 0.10, 0.20]
    labels = get_dataset_labels(base_train)
    scores_np = scores.detach().cpu().numpy()
    all_idx = np.arange(len(base_train))
    drop_curves = {"Drop helpful": [], "Drop harmful": [], "Drop random": []}
    keep_curves = {"Keep helpful": [], "Keep harmful": [], "Keep random": []}

    for ratio in drop_ratios:
        for name, mode in [("Drop helpful", "helpful"), ("Drop harmful", "harmful"), ("Drop random", "random")]:
            remove_idx = class_balanced_pick(labels, scores_np, ratio, mode, config.seed)
            keep_idx = np.setdiff1d(all_idx, remove_idx)
            acc = run_retrain_curve(Subset(base_train, keep_idx.tolist()), val_dataset, optimizer_name, config)
            drop_curves[name].append(acc)

    for ratio in keep_ratios:
        for name, mode in [("Keep helpful", "helpful"), ("Keep harmful", "harmful"), ("Keep random", "random")]:
            keep_idx = class_balanced_pick(labels, scores_np, ratio, mode, config.seed)
            acc = run_retrain_curve(Subset(base_train, keep_idx.tolist()), val_dataset, optimizer_name, config)
            keep_curves[name].append(acc)
    return drop_curves, keep_curves, drop_ratios, keep_ratios


def load_model_from_snapshot(snapshot: LocalStepSnapshot, hidden_dim: int) -> nn.Module:
    model = make_model(hidden_dim)
    model.load_state_dict({k: v.to(DEVICE) for k, v in snapshot.state_dict.items()})
    return model


def utility_of_subset_snapshot(snapshot: LocalStepSnapshot, subset_ids: List[int], config: RunConfig) -> float:
    """真实单步局部 utility，用于 MC Shapley；返回验证 loss 的变化量。"""
    loss_mean = make_eval_loss(config)
    model = load_model_from_snapshot(snapshot, config.hidden_dim)
    model.eval()
    val_imgs, val_labels = snapshot.val_imgs.to(DEVICE), snapshot.val_labels.to(DEVICE)
    with torch.no_grad():
        base_loss = loss_mean(model(val_imgs), val_labels).item()
    if len(subset_ids) == 0:
        return 0.0

    batch_imgs = snapshot.images[subset_ids].to(DEVICE)
    batch_labels = snapshot.labels[subset_ids].to(DEVICE)
    model.train(_UTILITY_TRAIN_MODE)
    zero_model_grads(model)
    if snapshot.optimizer_name == "sgd":
        logits = model(batch_imgs)
        if logits.dim() == 3:
            # 语言模型 logits 为 [batch, seq, vocab]；用注册的 LM loss，再按样本数近似求和。
            subset_loss = loss_mean(logits, batch_labels) * batch_imgs.size(0)
        else:
            subset_loss = nn.CrossEntropyLoss(reduction="sum")(logits, batch_labels)
        subset_loss.backward()
        step_scale = snapshot.lr / snapshot.batch_size_ref
        updated = load_model_from_snapshot(snapshot, config.hidden_dim)
        with torch.no_grad():
            for (_, p_u), (_, p_src) in zip(updated.named_parameters(), model.named_parameters()):
                if p_src.grad is not None:
                    p_u -= step_scale * p_src.grad
    else:
        subset_loss = loss_mean(model(batch_imgs), batch_labels)
        subset_loss.backward()
        updated = load_model_from_snapshot(snapshot, config.hidden_dim)
        if snapshot.adam_state is None:
            raise RuntimeError("Adam MC 验证需要 adam_state。")
        with torch.no_grad():
            for name, p_u in updated.named_parameters():
                p_src = dict(model.named_parameters())[name]
                if p_src.grad is None:
                    continue
                prev = snapshot.adam_state[name]
                prev_m = prev["exp_avg"].to(DEVICE)
                prev_v = prev["exp_avg_sq"].to(DEVICE)
                step_t = int(prev["step"]) + 1
                m_t = config.beta1 * prev_m + (1.0 - config.beta1) * p_src.grad
                v_t = config.beta2 * prev_v + (1.0 - config.beta2) * (p_src.grad * p_src.grad)
                m_hat = m_t / (1.0 - config.beta1 ** step_t)
                v_hat = v_t / (1.0 - config.beta2 ** step_t)
                p_u -= snapshot.lr * (m_hat / (torch.sqrt(v_hat) + config.eps))

    updated.eval()
    with torch.no_grad():
        new_loss = loss_mean(updated(val_imgs), val_labels).item()
    return new_loss - base_loss


def monte_carlo_local_shapley(snapshot: LocalStepSnapshot, config: RunConfig) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """用随机排列估计一个 step 内的局部 Shapley，作为一阶/二阶近似准确性的参照。"""
    mc_n = min(config.mc_batch_size, snapshot.images.size(0))
    ids = list(range(mc_n))
    values = np.zeros(mc_n, dtype=np.float64)
    for _ in range(config.mc_permutations):
        perm = ids.copy()
        random.shuffle(perm)
        prefix: List[int] = []
        prev_u = utility_of_subset_snapshot(snapshot, prefix, config)
        for idx in perm:
            new_prefix = prefix + [idx]
            new_u = utility_of_subset_snapshot(snapshot, new_prefix, config)
            values[idx] += new_u - prev_u
            prefix = new_prefix
            prev_u = new_u
    values /= max(config.mc_permutations, 1)
    approx = snapshot.approx_scores[:mc_n].astype(np.float64)
    topo_approx = None
    if snapshot.topo_approx_scores is not None:
        topo_approx = snapshot.topo_approx_scores[:mc_n].astype(np.float64)
    return values, approx, topo_approx


def corr_text(x: np.ndarray, y: np.ndarray) -> str:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return "Pearson=nan\nSpearman=nan"
    return f"Pearson={pearsonr(x, y)[0]:.3f}\nSpearman={spearmanr(x, y)[0]:.3f}"


def plot_sample_grid(ax, dataset: Dataset, sample_indices: np.ndarray, title: str, scores: np.ndarray) -> None:
    if _PLOT_SAMPLE_GRID is None:
        raise RuntimeError("尚未注册样本可视化函数，请先调用 configure_runtime(..., sample_plotter=...)")
    _PLOT_SAMPLE_GRID(ax, dataset, sample_indices, title, scores)


def make_summary_figure(
    plot_path: str,
    title: str,
    base_train: Dataset,
    scores: torch.Tensor,
    mc_values: np.ndarray,
    approx_values: np.ndarray,
    drop_curves: Dict[str, List[float]],
    keep_curves: Dict[str, List[float]],
    drop_ratios: List[float],
    keep_ratios: List[float],
    timing: Optional[Dict[str, float]] = None,
    topo_values: Optional[np.ndarray] = None,
) -> Path:
    """保存总图：近似准确性、删/留实验、好坏样本可视化、运行时间统计。"""
    out = Path(plot_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    scores_np = scores.detach().cpu().numpy()
    helpful = np.argsort(scores_np)[:10]
    harmful = np.argsort(scores_np)[-10:][::-1]

    fig = plt.figure(figsize=(16, 13))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.75])
    fig.suptitle(title, fontsize=14)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(mc_values, approx_values, s=34, alpha=0.85, label="Approx")
    if topo_values is not None:
        ax1.scatter(mc_values, topo_values, s=28, alpha=0.65, label="Graph score")
    mn = min(mc_values.min(), approx_values.min())
    mx = max(mc_values.max(), approx_values.max())
    if topo_values is not None:
        mn = min(mn, topo_values.min())
        mx = max(mx, topo_values.max())
    if mx > mn:
        ax1.plot([mn, mx], [mn, mx], "--", color="gray", linewidth=1)
    ax1.set_title("Local MC Shapley fidelity")
    ax1.set_xlabel("MC local Shapley")
    ax1.set_ylabel("In-Run approximation")
    text = corr_text(mc_values, approx_values)
    if topo_values is not None:
        text += "\nTopo " + corr_text(mc_values, topo_values).replace("\n", ", ")
    ax1.text(0.04, 0.96, text, transform=ax1.transAxes, va="top", bbox=dict(facecolor="white", alpha=0.85))
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(gs[0, 1])
    for name, vals in drop_curves.items():
        ax2.plot(np.array(drop_ratios) * 100, vals, marker="o", label=name)
    ax2.set_title("Drop helpful/harmful samples")
    ax2.set_xlabel("Removed ratio (%)")
    ax2.set_ylabel("Validation accuracy")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(gs[1, 0])
    for name, vals in keep_curves.items():
        ax3.plot(np.array(keep_ratios) * 100, vals, marker="o", label=name)
    ax3.set_title("Keep helpful/harmful samples")
    ax3.set_xlabel("Kept ratio (%)")
    ax3.set_ylabel("Validation accuracy")
    ax3.grid(alpha=0.3)
    ax3.legend(fontsize=8)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    lines = [
        "Sign: lower score = more helpful; higher score = more harmful.",
        f"Score mean: {scores_np.mean():.4g}",
        f"Score std: {scores_np.std():.4g}",
        f"MC samples: {len(mc_values)}",
    ]
    if timing:
        lines.append("")
        lines.append("Runtime:")
        for key, val in timing.items():
            lines.append(f"{key}: {val:.3f}s")
    ax4.text(0.02, 0.98, "\n".join(lines), va="top", fontsize=10)

    ax5 = fig.add_subplot(gs[2, 0])
    plot_sample_grid(ax5, base_train, helpful, "Most helpful samples", scores_np)
    ax6 = fig.add_subplot(gs[2, 1])
    plot_sample_grid(ax6, base_train, harmful, "Most harmful samples", scores_np)
    plt.tight_layout()
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def finish_experiment(
    title: str,
    config: RunConfig,
    optimizer_name: str,
    base_train: Dataset,
    val_dataset: Dataset,
    scores: torch.Tensor,
    snapshot: LocalStepSnapshot,
    timing: Optional[Dict[str, float]] = None,
) -> Path:
    """统一执行 MC 验证、删/留重训和总图保存。"""
    mc_values, approx_values, topo_values = monte_carlo_local_shapley(snapshot, config)
    drop_curves, keep_curves, drop_ratios, keep_ratios = run_drop_keep_experiments(
        base_train, val_dataset, scores, optimizer_name, config
    )
    plot_path = config.plot_output
    if not plot_path:
        plot_path = str(Path(config.output).with_suffix(".png"))
    return make_summary_figure(
        plot_path,
        title,
        base_train,
        scores,
        mc_values,
        approx_values,
        drop_curves,
        keep_curves,
        drop_ratios,
        keep_ratios,
        timing=timing,
        topo_values=topo_values,
    )



def _corr_pair(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), float("nan")
    return float(pearsonr(x, y)[0]), float(spearmanr(x, y)[0])


def finish_graph_comparison_experiment(
    title: str,
    config: RunConfig,
    base_train: Dataset,
    val_dataset: Dataset,
    methods: List[Dict],
    plot_path: str,
) -> Path:
    """Graph 实验专用总图：把 SGD、Adam-aware、Graph 方法的精度、性能和时间放在一起。"""
    out = Path(plot_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    drop_ratios_ref = None
    keep_ratios_ref = None
    results = []
    for method in methods:
        mc_values, approx_values, topo_values = monte_carlo_local_shapley(method["snapshot"], config)
        fidelity_values = topo_values if method.get("use_topo_fidelity", False) and topo_values is not None else approx_values
        pear, spear = _corr_pair(mc_values, fidelity_values)
        drop_curves, keep_curves, drop_ratios, keep_ratios = run_drop_keep_experiments(
            base_train, val_dataset, method["scores"], method["optimizer_name"], config
        )
        drop_ratios_ref = drop_ratios
        keep_ratios_ref = keep_ratios
        results.append(
            {
                "name": method["name"],
                "scores": method["scores"],
                "timing": method.get("timing", {}),
                "mc": mc_values,
                "approx": fidelity_values,
                "pearson": pear,
                "spearman": spear,
                "drop": drop_curves,
                "keep": keep_curves,
            }
        )

    graph_result = results[-1]
    graph_scores_np = graph_result["scores"].detach().cpu().numpy()
    graph_labels = get_dataset_labels(base_train)
    helpful_mean, harmful_mean = selected_score_means(graph_labels, graph_scores_np, 0.10, config.seed)
    helpful = np.argsort(graph_scores_np)[:10]
    harmful = np.argsort(graph_scores_np)[-10:][::-1]

    fig = plt.figure(figsize=(18, 15))
    gs = fig.add_gridspec(4, 2, height_ratios=[1.0, 1.0, 1.0, 1.65])
    fig.suptitle(title, fontsize=15)

    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(results))
    ax1.bar(x - 0.18, [r["pearson"] for r in results], width=0.36, label="Pearson")
    ax1.bar(x + 0.18, [r["spearman"] for r in results], width=0.36, label="Spearman")
    ax1.set_xticks(x)
    ax1.set_xticklabels([r["name"] for r in results], rotation=12, ha="right")
    ax1.set_ylim(-1.0, 1.0)
    ax1.set_title("Local MC fidelity comparison")
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(gs[0, 1])
    for r in results:
        ax2.scatter(r["mc"], r["approx"], s=28, alpha=0.75, label=r["name"])
    ax2.set_title("MC vs approximation")
    ax2.set_xlabel("MC local Shapley")
    ax2.set_ylabel("Approximation")
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(gs[1, 0])
    for r in results:
        ax3.plot(np.array(drop_ratios_ref) * 100, r["drop"]["Drop helpful"], marker="o", label=r["name"])
    ax3.set_title("Drop helpful samples")
    ax3.set_xlabel("Removed ratio (%)")
    ax3.set_ylabel("Validation accuracy")
    ax3.grid(alpha=0.3)
    ax3.legend(fontsize=8)

    ax4 = fig.add_subplot(gs[1, 1])
    for r in results:
        ax4.plot(np.array(keep_ratios_ref) * 100, r["keep"]["Keep helpful"], marker="o", label=r["name"])
    ax4.set_title("Keep helpful samples")
    ax4.set_xlabel("Kept ratio (%)")
    ax4.set_ylabel("Validation accuracy")
    ax4.grid(alpha=0.3)
    ax4.legend(fontsize=8)

    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(np.array(drop_ratios_ref) * 100, graph_result["drop"]["Drop helpful"], marker="o", label="Drop helpful")
    ax5.plot(np.array(drop_ratios_ref) * 100, graph_result["drop"]["Drop harmful"], marker="o", label="Drop harmful")
    ax5.plot(np.array(drop_ratios_ref) * 100, graph_result["drop"]["Drop random"], marker="o", linestyle="--", label="Drop random")
    ax5.set_title("Graph method drop sanity check")
    ax5.text(
        0.02,
        0.92,
        f"10% selected score mean: helpful={helpful_mean:.3g}, harmful={harmful_mean:.3g}",
        transform=ax5.transAxes,
        fontsize=8,
        va="top",
        bbox=dict(facecolor="white", alpha=0.8),
    )
    ax5.set_xlabel("Removed ratio (%)")
    ax5.set_ylabel("Validation accuracy")
    ax5.grid(alpha=0.3)
    ax5.legend(fontsize=8)

    ax6 = fig.add_subplot(gs[2, 1])
    graph_timing = graph_result["timing"]
    score_times = [r["timing"].get("Score time", 0.0) for r in results]
    second_order_times = [
        r["timing"].get("Second-order component time", r["timing"].get("HVP second-order time", 0.0))
        for r in results
    ]
    graph_build_time = graph_timing.get("Graph build time", 0.0)
    graph_raw_time = graph_timing.get("Raw Adam score time", 0.0)
    graph_topology_time = graph_timing.get("Topology distance time", 0.0)
    graph_train_time = graph_timing.get("Train update time", 0.0)

    width = 0.36
    ax6.bar(x - width / 2, score_times, width=width, label="Online pass time (raw + second + train)")
    ax6.bar(x + width / 2, second_order_times, width=width, label="Second-order component only")
    ax6.set_xticks(x)
    ax6.set_xticklabels([r["name"] for r in results], rotation=12, ha="right")
    ax6.set_title("Runtime: online pass vs second-order only")
    ax6.set_ylabel("Seconds")
    ax6.grid(axis="y", alpha=0.25)
    for xpos, total_val, second_val in zip(x, score_times, second_order_times):
        ax6.text(xpos - width / 2, total_val, f"{total_val:.2f}s", ha="center", va="bottom", fontsize=7)
        ax6.text(xpos + width / 2, second_val, f"{second_val:.3f}s", ha="center", va="bottom", fontsize=7)
    ax6.text(
        0.02,
        0.95,
        (
            f"Graph build: {graph_build_time:.2f}s offline/cacheable\n"
            f"Graph raw Adam: {graph_raw_time:.2f}s; topology distance: {graph_topology_time:.4f}s; "
            f"train/misc: {graph_train_time:.2f}s"
        ),
        transform=ax6.transAxes,
        va="top",
        fontsize=8,
        bbox=dict(facecolor="white", alpha=0.85),
    )
    ax6.legend(fontsize=7, loc="upper right")

    ax7 = fig.add_subplot(gs[3, 0])
    plot_sample_grid(ax7, base_train, helpful, "Graph: most helpful samples", graph_scores_np)
    ax8 = fig.add_subplot(gs[3, 1])
    plot_sample_grid(ax8, base_train, harmful, "Graph: most harmful samples", graph_scores_np)

    plt.tight_layout()
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out
