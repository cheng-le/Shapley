import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


# =========================================================
# Config
# =========================================================
SEED = 0
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 64
EPOCHS_SCORE = 1
EPOCHS_PRUNE = 3
LR = 0.01
MC_BATCH_SIZE = 12
MC_PERMUTATIONS = 200
DROP_RATIOS = [0.01, 0.05, 0.10]
KEEP_RATIOS = [0.05, 0.10, 0.20]
VAL_MC_SAMPLES = 4
PLOT_PATH = 'mnist_sgd_eval.png'
TOPK_VIS = 10


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================
# Dataset helpers
# =========================================================
class IndexedDataset(Dataset):
    def __init__(self, base_dataset: Dataset):
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        x, y = self.base_dataset[idx]
        return x, y, idx


# =========================================================
# Model
# =========================================================
class MLP_MNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# =========================================================
# Training / eval helpers
# =========================================================
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss = loss_fn(logits, y)
        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_count += x.size(0)
    return total_loss / total_count, total_correct / total_count


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer, loss_fn: nn.Module) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_count += x.size(0)
    return total_loss / total_count


def get_grad_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    grads: Dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            grads[name] = p.grad.detach().clone()
    return grads


def zero_model_grads(model: nn.Module) -> None:
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()


# =========================================================
# In-Run first-order score computation + capture one step
# =========================================================
@dataclass
class StepSnapshot:
    state_dict: Dict[str, torch.Tensor]
    images: torch.Tensor
    labels: torch.Tensor
    indices: torch.Tensor
    val_imgs: torch.Tensor
    val_labels: torch.Tensor
    approx_scores: np.ndarray
    lr: float
    batch_size_ref: int


def compute_inrun_scores_and_capture_step(
    train_loader: DataLoader,
    val_reference_loader: DataLoader,
    capture_step: int = 0,
) -> Tuple[torch.Tensor, StepSnapshot]:
    """
    Compute first-order SGD-style In-Run scores for one pass over the train set,
    and capture one specific step for local MC fidelity validation.

    IMPORTANT: the analytic score is matched to the real SGD update induced by
    CrossEntropyLoss(reduction='mean'), namely:
        w_{t+1} = w_t - lr * (1/B) * sum_i g_i
    hence score_i = -(lr / B) * <g_i, g_val>.
    """
    model = MLP_MNIST().to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    data_scores = torch.zeros(len(train_loader.dataset), dtype=torch.float32, device='cpu')

    val_iter = iter(val_reference_loader)
    val_imgs_list, val_labels_list = [], []
    for _ in range(VAL_MC_SAMPLES):
        try:
            vx, vy = next(val_iter)
        except StopIteration:
            val_iter = iter(val_reference_loader)
            vx, vy = next(val_iter)
        val_imgs_list.append(vx)
        val_labels_list.append(vy)
    val_imgs = torch.cat(val_imgs_list, dim=0).to(DEVICE)
    val_labels = torch.cat(val_labels_list, dim=0).to(DEVICE)

    captured_snapshot = None
    global_step = 0

    for _ in range(EPOCHS_SCORE):
        model.train()
        for images, labels, indices in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            indices = indices.to(DEVICE)
            batch_size_ref = images.size(0)

            optimizer.zero_grad()
            logits_val_only = model(val_imgs)
            loss_val_only = loss_fn(logits_val_only, val_labels)
            loss_val_only.backward()
            grad_val = get_grad_dict(model)

            lr_now = optimizer.param_groups[0]['lr']
            approx_scores = []
            for i in range(batch_size_ref):
                zero_model_grads(model)
                single_logit = model(images[i].unsqueeze(0))
                single_loss = loss_fn(single_logit, labels[i].unsqueeze(0))
                single_loss.backward()

                dot_val = 0.0
                for name, p in model.named_parameters():
                    if p.grad is not None and name in grad_val:
                        dot_val += (p.grad.view(-1) * grad_val[name].view(-1)).sum().item()
                score_i = -(lr_now / batch_size_ref) * dot_val
                approx_scores.append(score_i)
                data_scores[indices[i].item()] += score_i

            if global_step == capture_step:
                captured_snapshot = StepSnapshot(
                    state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    images=images.detach().cpu().clone(),
                    labels=labels.detach().cpu().clone(),
                    indices=indices.detach().cpu().clone(),
                    val_imgs=val_imgs.detach().cpu().clone(),
                    val_labels=val_labels.detach().cpu().clone(),
                    approx_scores=np.array(approx_scores, dtype=np.float64),
                    lr=lr_now,
                    batch_size_ref=batch_size_ref,
                )

            optimizer.zero_grad()
            logits = model(images)
            loss_train = loss_fn(logits, labels)
            loss_train.backward()
            optimizer.step()
            global_step += 1

    if captured_snapshot is None:
        raise RuntimeError('Failed to capture a training step for MC validation.')

    return data_scores, captured_snapshot


# =========================================================
# Local MC In-Run Shapley for one step
# =========================================================
def utility_of_subset(
    base_state_dict: Dict[str, torch.Tensor],
    subset_local_ids: List[int],
    batch_imgs: torch.Tensor,
    batch_labels: torch.Tensor,
    val_imgs: torch.Tensor,
    val_labels: torch.Tensor,
    lr: float,
    batch_size_ref: int,
    loss_fn_eval: nn.Module,
) -> float:
    """
    U^(t)(S) = loss(w_tilde(S), val) - loss(w_t, val)

    To match the actual training step with CrossEntropyLoss(reduction='mean') on a
    full batch of size B, we update with:
        w_tilde(S) = w_t - (lr / B) * sum_{i in S} g_i
    so subset gradients must be computed with SUM reduction, not MEAN reduction.
    """
    model = MLP_MNIST().to(DEVICE)
    model.load_state_dict({k: v.to(DEVICE) for k, v in base_state_dict.items()})
    loss_fn_sum = nn.CrossEntropyLoss(reduction='sum')

    model.eval()
    with torch.no_grad():
        baseline_loss = loss_fn_eval(model(val_imgs), val_labels).item()

    if len(subset_local_ids) == 0:
        return 0.0

    model.train()
    zero_model_grads(model)
    subset_x = batch_imgs[subset_local_ids].to(DEVICE)
    subset_y = batch_labels[subset_local_ids].to(DEVICE)
    subset_loss = loss_fn_sum(model(subset_x), subset_y)
    subset_loss.backward()

    updated_model = MLP_MNIST().to(DEVICE)
    updated_model.load_state_dict({k: v.to(DEVICE) for k, v in base_state_dict.items()})
    step_lr = lr / batch_size_ref
    with torch.no_grad():
        for (_, p_u), (_, p) in zip(updated_model.named_parameters(), model.named_parameters()):
            if p.grad is not None:
                p_u -= step_lr * p.grad

    updated_model.eval()
    with torch.no_grad():
        updated_loss = loss_fn_eval(updated_model(val_imgs), val_labels).item()
    return updated_loss - baseline_loss


def monte_carlo_local_shapley(snapshot: StepSnapshot, mc_batch_size: int, num_perm: int) -> Tuple[np.ndarray, np.ndarray]:
    loss_fn = nn.CrossEntropyLoss()

    batch_size = snapshot.images.size(0)
    mc_batch_size = min(mc_batch_size, batch_size)
    local_ids = list(range(mc_batch_size))
    mc_values = np.zeros(mc_batch_size, dtype=np.float64)

    batch_imgs = snapshot.images
    batch_labels = snapshot.labels
    val_imgs = snapshot.val_imgs.to(DEVICE)
    val_labels = snapshot.val_labels.to(DEVICE)

    for _ in range(num_perm):
        perm = local_ids.copy()
        random.shuffle(perm)
        prefix: List[int] = []
        prev_u = utility_of_subset(
            snapshot.state_dict, prefix, batch_imgs, batch_labels,
            val_imgs, val_labels, snapshot.lr, snapshot.batch_size_ref, loss_fn
        )
        for idx in perm:
            new_prefix = prefix + [idx]
            new_u = utility_of_subset(
                snapshot.state_dict, new_prefix, batch_imgs, batch_labels,
                val_imgs, val_labels, snapshot.lr, snapshot.batch_size_ref, loss_fn
            )
            mc_values[idx] += new_u - prev_u
            prefix = new_prefix
            prev_u = new_u

    mc_values /= num_perm
    approx_values = snapshot.approx_scores[:mc_batch_size].astype(np.float64)
    return mc_values, approx_values


# =========================================================
# Pruning experiment (class-balanced)
# =========================================================
def get_dataset_labels(dataset: Dataset) -> np.ndarray:
    if hasattr(dataset, 'targets'):
        targets = dataset.targets
        if isinstance(targets, torch.Tensor):
            return targets.cpu().numpy().astype(int)
        return np.asarray(targets, dtype=int)
    labels = []
    for _, y in dataset:
        labels.append(int(y))
    return np.asarray(labels, dtype=int)


def class_balanced_remove_indices(labels: np.ndarray, scores: np.ndarray, ratio: float, mode: str, seed: int) -> np.ndarray:
    """
    mode in {'most_helpful','most_harmful','random'}
    Helpful = most negative scores, harmful = most positive scores.
    """
    rng = np.random.default_rng(seed)
    remove = []
    classes = np.unique(labels)
    for c in classes:
        idx_c = np.where(labels == c)[0]
        remove_n = int(round(len(idx_c) * ratio))
        if remove_n <= 0:
            continue
        if mode == 'most_helpful':
            chosen = idx_c[np.argsort(scores[idx_c])[:remove_n]]
        elif mode == 'most_harmful':
            chosen = idx_c[np.argsort(scores[idx_c])[-remove_n:]]
        elif mode == 'random':
            chosen = rng.choice(idx_c, size=remove_n, replace=False)
        else:
            raise ValueError(mode)
        remove.append(chosen)
    if not remove:
        return np.array([], dtype=int)
    return np.concatenate(remove)


def class_balanced_keep_indices(labels: np.ndarray, scores: np.ndarray, ratio: float, mode: str, seed: int) -> np.ndarray:
    """
    mode in {'most_helpful','most_harmful','random'}
    ratio is kept fraction.
    """
    rng = np.random.default_rng(seed)
    keep = []
    classes = np.unique(labels)
    for c in classes:
        idx_c = np.where(labels == c)[0]
        keep_n = max(1, int(round(len(idx_c) * ratio)))
        if mode == 'most_helpful':
            chosen = idx_c[np.argsort(scores[idx_c])[:keep_n]]
        elif mode == 'most_harmful':
            chosen = idx_c[np.argsort(scores[idx_c])[-keep_n:]]
        elif mode == 'random':
            chosen = rng.choice(idx_c, size=keep_n, replace=False)
        else:
            raise ValueError(mode)
        keep.append(chosen)
    return np.concatenate(keep)


def train_and_record_curves(train_dataset: Dataset, val_dataset: Dataset, epochs: int) -> Tuple[List[float], List[float]]:
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    model = MLP_MNIST().to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    val_losses, val_accs = [], []
    for _ in range(epochs):
        train_one_epoch(model, train_loader, optimizer, loss_fn)
        v_loss, v_acc = evaluate(model, val_loader, loss_fn)
        val_losses.append(v_loss)
        val_accs.append(v_acc)
    return val_losses, val_accs


def run_drop_curves(base_train_dataset: Dataset, val_dataset: Dataset, scores: torch.Tensor) -> Dict[str, List[float]]:
    labels = get_dataset_labels(base_train_dataset)
    scores_np = scores.numpy()
    acc_curves = {
        'Drop most helpful': [],
        'Drop most harmful': [],
        'Drop random': [],
    }
    for ratio in DROP_RATIOS:
        for name, mode in [
            ('Drop most helpful', 'most_helpful'),
            ('Drop most harmful', 'most_harmful'),
            ('Drop random', 'random'),
        ]:
            remove_idx = class_balanced_remove_indices(labels, scores_np, ratio, mode, SEED)
            keep_idx = np.setdiff1d(np.arange(len(base_train_dataset)), remove_idx)
            subset = Subset(base_train_dataset, keep_idx.tolist())
            _, accs = train_and_record_curves(subset, val_dataset, EPOCHS_PRUNE)
            acc_curves[name].append(accs[-1])
    return acc_curves


def run_keep_curves(base_train_dataset: Dataset, val_dataset: Dataset, scores: torch.Tensor) -> Dict[str, List[float]]:
    labels = get_dataset_labels(base_train_dataset)
    scores_np = scores.numpy()
    acc_curves = {
        'Keep most helpful': [],
        'Keep most harmful': [],
        'Keep random': [],
    }
    for ratio in KEEP_RATIOS:
        for name, mode in [
            ('Keep most helpful', 'most_helpful'),
            ('Keep most harmful', 'most_harmful'),
            ('Keep random', 'random'),
        ]:
            keep_idx = class_balanced_keep_indices(labels, scores_np, ratio, mode, SEED)
            subset = Subset(base_train_dataset, keep_idx.tolist())
            _, accs = train_and_record_curves(subset, val_dataset, EPOCHS_PRUNE)
            acc_curves[name].append(accs[-1])
    return acc_curves


# =========================================================
# Visualization
# =========================================================
def plot_sample_grid(ax, base_train_dataset: Dataset, sample_indices: np.ndarray, title: str, scores: np.ndarray) -> None:
    ax.set_title(title, fontsize=12)
    ax.axis('off')

    n = len(sample_indices)
    cols = 5
    rows = int(np.ceil(n / cols))

    bbox = ax.get_position()
    fig = ax.figure
    pad_x, pad_y = 0.005, 0.02
    cell_w = (bbox.width - pad_x * (cols - 1)) / cols
    cell_h = (bbox.height - pad_y * (rows - 1)) / rows

    for k, idx in enumerate(sample_indices):
        r = k // cols
        c = k % cols
        left = bbox.x0 + c * (cell_w + pad_x)
        # bottom = bbox.y1 - (r + 1) * cell_h - r * pad_y
        bottom = bbox.y1 - (r + 1) * cell_h - r * pad_y - 0.09
        subax = fig.add_axes([left, bottom, cell_w, cell_h])
        img, y = base_train_dataset[idx]
        subax.imshow(img.squeeze(0).numpy(), cmap='gray')
        subax.set_title(f'i={idx}\ny={y}\n{scores[idx]:.3f}', fontsize=8)
        subax.axis('off')


def make_summary_figure(
    mc_values: np.ndarray,
    approx_values: np.ndarray,
    drop_curves: Dict[str, List[float]],
    keep_curves: Dict[str, List[float]],
    base_train_dataset: Dataset,
    scores: torch.Tensor,
    save_path: str,
) -> None:
    pear = pearsonr(mc_values, approx_values)[0]
    spear = spearmanr(mc_values, approx_values)[0]
    scores_np = scores.numpy()

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.5])

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(mc_values, approx_values, s=40, alpha=0.85)
    mn = min(mc_values.min(), approx_values.min())
    mx = max(mc_values.max(), approx_values.max())
    if mx > mn:
        ax1.plot([mn, mx], [mn, mx], '--', color='gray', linewidth=1)
    ax1.set_title('Single-step MC Fidelity')
    ax1.set_xlabel('MC local Shapley')
    ax1.set_ylabel('First-order In-Run approx')
    ax1.text(
        0.05, 0.95,
        f'Pearson = {pear:.3f}\nSpearman = {spear:.3f}\nN = {len(mc_values)}',
        transform=ax1.transAxes,
        va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85)
    )

    ax2 = fig.add_subplot(gs[0, 1])
    for name, vals in drop_curves.items():
        ax2.plot(np.array(DROP_RATIOS) * 100, vals, marker='o', label=name)
    ax2.set_title('Class-balanced drop-based pruning')
    ax2.set_xlabel('Removed ratio (%)')
    ax2.set_ylabel(f'Final val acc after {EPOCHS_PRUNE} epochs')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 0])
    for name, vals in keep_curves.items():
        ax3.plot(np.array(KEEP_RATIOS) * 100, vals, marker='o', label=name)
    ax3.set_title('Class-balanced keep-only comparison')
    ax3.set_xlabel('Kept ratio (%)')
    ax3.set_ylabel(f'Final val acc after {EPOCHS_PRUNE} epochs')
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)

    # small textual summary panel
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    summary_lines = [
        'Sign convention:',
        '  more negative score = more helpful',
        '  more positive score = more harmful / less helpful',
        '',
        'MC utility matched to training step:',
        '  U(S) = loss(w_tilde(S), val) - loss(w_t, val)',
        '  w_tilde(S) = w_t - (lr / B) * sum_{i in S} g_i',
        '',
        f'VAL_MC_SAMPLES = {VAL_MC_SAMPLES}',
        f'MC_BATCH_SIZE = {MC_BATCH_SIZE}',
        f'MC_PERMUTATIONS = {MC_PERMUTATIONS}',
    ]
    ax4.text(0.02, 0.98, '\n'.join(summary_lines), va='top', fontsize=10)

    helpful_idx = np.argsort(scores_np)[:TOPK_VIS]
    harmful_idx = np.argsort(scores_np)[-TOPK_VIS:]

    ax5 = fig.add_subplot(gs[2, 0])
    plot_sample_grid(ax5, base_train_dataset, helpful_idx, 'Most helpful samples (most negative scores)', scores_np)

    ax6 = fig.add_subplot(gs[2, 1])
    plot_sample_grid(ax6, base_train_dataset, harmful_idx[::-1], 'Most harmful samples (most positive scores)', scores_np)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# =========================================================
# Main
# =========================================================
def main() -> None:
    set_seed(SEED)

    transform = transforms.ToTensor()
    base_train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, transform=transform, download=True
    )
    val_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, transform=transform, download=True
    )

    indexed_train_dataset = IndexedDataset(base_train_dataset)
    train_loader = DataLoader(indexed_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_reference_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    print('Step 1/4: computing first-order In-Run scores and capturing one step ...')
    data_scores, snapshot = compute_inrun_scores_and_capture_step(
        train_loader=train_loader,
        val_reference_loader=val_reference_loader,
        capture_step=0,
    )

    print('Step 2/4: running single-step MC fidelity validation ...')
    mc_values, approx_values = monte_carlo_local_shapley(
        snapshot=snapshot,
        mc_batch_size=MC_BATCH_SIZE,
        num_perm=MC_PERMUTATIONS,
    )

    print('Step 3/4: running class-balanced pruning comparisons ...')
    drop_curves = run_drop_curves(base_train_dataset, val_dataset, data_scores)
    keep_curves = run_keep_curves(base_train_dataset, val_dataset, data_scores)

    print('Step 4/4: saving summary figure ...')
    make_summary_figure(
        mc_values=mc_values,
        approx_values=approx_values,
        drop_curves=drop_curves,
        keep_curves=keep_curves,
        base_train_dataset=base_train_dataset,
        scores=data_scores,
        save_path=PLOT_PATH,
    )

    pear = pearsonr(mc_values, approx_values)[0]
    spear = spearmanr(mc_values, approx_values)[0]
    print(f'Pearson:  {pear:.4f}')
    print(f'Spearman: {spear:.4f}')
    print(f'Figure saved to: {PLOT_PATH}')


if __name__ == '__main__':
    main()
