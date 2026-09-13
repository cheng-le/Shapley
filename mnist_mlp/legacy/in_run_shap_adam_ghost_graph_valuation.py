import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
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
LR = 1e-3
BETAS = (0.9, 0.999)
EPS = 1e-8
MC_BATCH_SIZE = 12
MC_PERMUTATIONS = 200
DROP_RATIOS = [0.01, 0.05, 0.10]
KEEP_RATIOS = [0.05, 0.10, 0.20]
VAL_MC_SAMPLES = 4
TOPK_VIS = 10
PLOT_PATH = 'mnist_topoaware_adam_eval.png'
TOPO_K = 10
TOPO_PCA_DIM = 64
ALPHA_TOPO = 0.2
LAMBDA_TOPO = 0.05


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class IndexedDataset(Dataset):
    def __init__(self, base_dataset: Dataset):
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        x, y = self.base_dataset[idx]
        return x, y, idx


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


def zero_model_grads(model: nn.Module) -> None:
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()


def get_grad_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    grads: Dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            grads[name] = p.grad.detach().clone()
    return grads


def safe_step_int(step_obj) -> int:
    if isinstance(step_obj, torch.Tensor):
        return int(step_obj.item())
    return int(step_obj)


def clone_adam_state(model: nn.Module, optimizer: optim.Optimizer) -> Dict[str, Dict[str, torch.Tensor | int]]:
    state_dict = {}
    for name, p in model.named_parameters():
        state = optimizer.state[p]
        if len(state) == 0:
            exp_avg = torch.zeros_like(p, device=p.device)
            exp_avg_sq = torch.zeros_like(p, device=p.device)
            step = 0
        else:
            exp_avg = state['exp_avg'].detach().clone()
            exp_avg_sq = state['exp_avg_sq'].detach().clone()
            step = safe_step_int(state['step'])
        state_dict[name] = {'exp_avg': exp_avg, 'exp_avg_sq': exp_avg_sq, 'step': step}
    return state_dict


def adam_update_direction_from_single_grad(sample_grad, adam_state_snapshot, beta1, beta2, eps):
    update_dir = {}
    for name, g in sample_grad.items():
        exp_avg_prev = adam_state_snapshot[name]['exp_avg']
        exp_avg_sq_prev = adam_state_snapshot[name]['exp_avg_sq']
        prev_step = int(adam_state_snapshot[name]['step'])
        step_t = prev_step + 1
        m_t = beta1 * exp_avg_prev + (1.0 - beta1) * g
        v_t = beta2 * exp_avg_sq_prev + (1.0 - beta2) * (g * g)
        bc1 = 1.0 - beta1 ** step_t
        bc2 = 1.0 - beta2 ** step_t
        m_hat = m_t / bc1
        v_hat = v_t / bc2
        update_dir[name] = m_hat / (torch.sqrt(v_hat) + eps)
    return update_dir


def dot_with_val_grad(update_dir, grad_val) -> float:
    score = 0.0
    for name in update_dir.keys():
        score += (update_dir[name].view(-1) * grad_val[name].view(-1)).sum().item()
    return score


@dataclass
class AdamStepSnapshot:
    state_dict: Dict[str, torch.Tensor]
    adam_state_snapshot: Dict[str, Dict[str, torch.Tensor | int]]
    images: torch.Tensor
    labels: torch.Tensor
    indices: torch.Tensor
    val_imgs: torch.Tensor
    val_labels: torch.Tensor
    approx_scores: np.ndarray
    lr: float


def build_knn_graph_from_raw_pixels(base_dataset: Dataset, k: int, pca_dim: int) -> sp.csr_matrix:
    print('Building offline KNN graph ...')
    X = np.stack([base_dataset[i][0].view(-1).numpy() for i in range(len(base_dataset))]).astype(np.float32)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    if pca_dim is not None and pca_dim < X.shape[1]:
        X = PCA(n_components=pca_dim, random_state=SEED).fit_transform(X).astype(np.float32)
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric='cosine', n_jobs=-1)
    nbrs.fit(X)
    dist, ind = nbrs.kneighbors(X)
    rows = np.repeat(np.arange(X.shape[0]), k)
    cols = ind[:, 1:].reshape(-1)
    sims = 1.0 - dist[:, 1:].reshape(-1)
    sims = np.clip(sims, 0.0, None)
    A = sp.coo_matrix((sims, (rows, cols)), shape=(X.shape[0], X.shape[0]))
    A = 0.5 * (A + A.T)
    A.setdiag(0.0)
    A.eliminate_zeros()
    row_sum = np.asarray(A.sum(axis=1)).ravel()
    row_sum[row_sum == 0] = 1.0
    A = sp.diags(1.0 / row_sum) @ A
    print('KNN graph built.')
    return A.tocsr()


def topology_regularize_batch_scores(batch_indices_np, raw_batch_scores_np, raw_score_buffer_np, A_csr, alpha, lambda_reg):
    tmp_raw = raw_score_buffer_np.copy()
    tmp_raw[batch_indices_np] += raw_batch_scores_np
    diffusion_batch = A_csr[batch_indices_np].dot(tmp_raw)
    crowding_batch = np.asarray(A_csr[batch_indices_np][:, batch_indices_np].sum(axis=1)).ravel()
    topo_batch_scores_np = raw_batch_scores_np + alpha * diffusion_batch - lambda_reg * crowding_batch
    return topo_batch_scores_np, diffusion_batch, crowding_batch


def compute_topo_adamaware_scores_and_capture_step(train_loader, val_reference_loader, base_train_dataset, A_csr, capture_step=0):
    model = MLP_MNIST().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, betas=BETAS, eps=EPS)
    loss_fn = nn.CrossEntropyLoss()
    raw_score_buffer = torch.zeros(len(train_loader.dataset), dtype=torch.float32, device='cpu')
    topo_score_buffer = torch.zeros(len(train_loader.dataset), dtype=torch.float32, device='cpu')

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
    beta1, beta2 = BETAS

    for _ in range(EPOCHS_SCORE):
        model.train()
        for images, labels, indices in train_loader:
            images, labels, indices = images.to(DEVICE), labels.to(DEVICE), indices.to(DEVICE)
            adam_state_snapshot = clone_adam_state(model, optimizer)

            optimizer.zero_grad()
            logits_val = model(val_imgs)
            loss_val = loss_fn(logits_val, val_labels)
            loss_val.backward()
            grad_val = get_grad_dict(model)

            lr_now = optimizer.param_groups[0]['lr']
            raw_batch_scores = []
            for i in range(images.size(0)):
                zero_model_grads(model)
                single_logit = model(images[i].unsqueeze(0))
                single_loss = loss_fn(single_logit, labels[i].unsqueeze(0))
                single_loss.backward()
                g_i = get_grad_dict(model)
                update_dir_i = adam_update_direction_from_single_grad(g_i, adam_state_snapshot, beta1, beta2, EPS)
                score_i = -lr_now * dot_with_val_grad(update_dir_i, grad_val)
                raw_batch_scores.append(score_i)
                raw_score_buffer[indices[i].item()] += score_i

            raw_batch_scores_np = np.asarray(raw_batch_scores, dtype=np.float32)
            batch_indices_np = indices.detach().cpu().numpy()
            topo_batch_scores_np, _, _ = topology_regularize_batch_scores(
                batch_indices_np, raw_batch_scores_np, raw_score_buffer.numpy(), A_csr, ALPHA_TOPO, LAMBDA_TOPO
            )
            topo_score_buffer[batch_indices_np] += torch.from_numpy(topo_batch_scores_np.astype(np.float32))

            if global_step == capture_step:
                captured_snapshot = AdamStepSnapshot(
                    state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    adam_state_snapshot={
                        k: {
                            'exp_avg': v['exp_avg'].detach().cpu().clone(),
                            'exp_avg_sq': v['exp_avg_sq'].detach().cpu().clone(),
                            'step': int(v['step']),
                        }
                        for k, v in adam_state_snapshot.items()
                    },
                    images=images.detach().cpu().clone(),
                    labels=labels.detach().cpu().clone(),
                    indices=indices.detach().cpu().clone(),
                    val_imgs=val_imgs.detach().cpu().clone(),
                    val_labels=val_labels.detach().cpu().clone(),
                    approx_scores=np.asarray(raw_batch_scores, dtype=np.float64),
                    lr=lr_now,
                )

            optimizer.zero_grad()
            logits = model(images)
            loss_train = loss_fn(logits, labels)
            loss_train.backward()
            optimizer.step()
            global_step += 1

    if captured_snapshot is None:
        raise RuntimeError('Failed to capture a training step for MC validation.')
    return raw_score_buffer, topo_score_buffer, captured_snapshot


def utility_of_subset_adam(base_state_dict, adam_state_snapshot, subset_local_ids, batch_imgs, batch_labels, val_imgs, val_labels, lr, loss_fn_eval):
    model = MLP_MNIST().to(DEVICE)
    model.load_state_dict({k: v.to(DEVICE) for k, v in base_state_dict.items()})
    model.eval()
    with torch.no_grad():
        baseline_loss = loss_fn_eval(model(val_imgs), val_labels).item()
    if len(subset_local_ids) == 0:
        return 0.0
    model.train()
    zero_model_grads(model)
    subset_x = batch_imgs[subset_local_ids].to(DEVICE)
    subset_y = batch_labels[subset_local_ids].to(DEVICE)
    subset_loss = loss_fn_eval(model(subset_x), subset_y)
    subset_loss.backward()
    updated_model = MLP_MNIST().to(DEVICE)
    updated_model.load_state_dict({k: v.to(DEVICE) for k, v in base_state_dict.items()})
    beta1, beta2 = BETAS
    with torch.no_grad():
        for (name_u, p_u), (_, p) in zip(updated_model.named_parameters(), model.named_parameters()):
            if p.grad is None:
                continue
            prev = adam_state_snapshot[name_u]
            exp_avg_prev = prev['exp_avg'].to(DEVICE)
            exp_avg_sq_prev = prev['exp_avg_sq'].to(DEVICE)
            prev_step = int(prev['step'])
            g = p.grad
            step_t = prev_step + 1
            m_t = beta1 * exp_avg_prev + (1.0 - beta1) * g
            v_t = beta2 * exp_avg_sq_prev + (1.0 - beta2) * (g * g)
            bc1 = 1.0 - beta1 ** step_t
            bc2 = 1.0 - beta2 ** step_t
            m_hat = m_t / bc1
            v_hat = v_t / bc2
            p_u -= lr * (m_hat / (torch.sqrt(v_hat) + EPS))
    updated_model.eval()
    with torch.no_grad():
        updated_loss = loss_fn_eval(updated_model(val_imgs), val_labels).item()
    return updated_loss - baseline_loss


def monte_carlo_local_shapley_adam(snapshot: AdamStepSnapshot, mc_batch_size: int, num_perm: int):
    loss_fn = nn.CrossEntropyLoss()
    batch_size = snapshot.images.size(0)
    mc_batch_size = min(mc_batch_size, batch_size)
    local_ids = list(range(mc_batch_size))
    mc_values = np.zeros(mc_batch_size, dtype=np.float64)
    batch_imgs, batch_labels = snapshot.images, snapshot.labels
    val_imgs, val_labels = snapshot.val_imgs.to(DEVICE), snapshot.val_labels.to(DEVICE)
    for _ in range(num_perm):
        perm = local_ids.copy()
        random.shuffle(perm)
        prefix: List[int] = []
        prev_u = utility_of_subset_adam(snapshot.state_dict, snapshot.adam_state_snapshot, prefix, batch_imgs, batch_labels, val_imgs, val_labels, snapshot.lr, loss_fn)
        for idx in perm:
            new_prefix = prefix + [idx]
            new_u = utility_of_subset_adam(snapshot.state_dict, snapshot.adam_state_snapshot, new_prefix, batch_imgs, batch_labels, val_imgs, val_labels, snapshot.lr, loss_fn)
            mc_values[idx] += new_u - prev_u
            prefix = new_prefix
            prev_u = new_u
    mc_values /= num_perm
    approx_values = snapshot.approx_scores[:mc_batch_size].astype(np.float64)
    return mc_values, approx_values


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


def class_balanced_remove_indices(labels, scores, ratio, mode, seed):
    rng = np.random.default_rng(seed)
    remove = []
    for c in np.unique(labels):
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
    return np.concatenate(remove) if remove else np.array([], dtype=int)


def class_balanced_keep_indices(labels, scores, ratio, mode, seed):
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(labels):
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


def train_and_record_curves(train_dataset: Dataset, val_dataset: Dataset, epochs: int):
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    model = MLP_MNIST().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, betas=BETAS, eps=EPS)
    loss_fn = nn.CrossEntropyLoss()
    val_losses, val_accs = [], []
    for _ in range(epochs):
        train_one_epoch(model, train_loader, optimizer, loss_fn)
        v_loss, v_acc = evaluate(model, val_loader, loss_fn)
        val_losses.append(v_loss)
        val_accs.append(v_acc)
    return val_losses, val_accs


def run_drop_curves(base_train_dataset, val_dataset, scores):
    labels = get_dataset_labels(base_train_dataset)
    scores_np = scores.numpy()
    acc_curves = {'Drop most helpful': [], 'Drop most harmful': [], 'Drop random': []}
    for ratio in DROP_RATIOS:
        for name, mode in [('Drop most helpful', 'most_helpful'), ('Drop most harmful', 'most_harmful'), ('Drop random', 'random')]:
            remove_idx = class_balanced_remove_indices(labels, scores_np, ratio, mode, SEED)
            keep_idx = np.setdiff1d(np.arange(len(base_train_dataset)), remove_idx)
            subset = Subset(base_train_dataset, keep_idx.tolist())
            _, accs = train_and_record_curves(subset, val_dataset, EPOCHS_PRUNE)
            acc_curves[name].append(accs[-1])
    return acc_curves


def run_keep_curves(base_train_dataset, val_dataset, scores):
    labels = get_dataset_labels(base_train_dataset)
    scores_np = scores.numpy()
    acc_curves = {'Keep most helpful': [], 'Keep most harmful': [], 'Keep random': []}
    for ratio in KEEP_RATIOS:
        for name, mode in [('Keep most helpful', 'most_helpful'), ('Keep most harmful', 'most_harmful'), ('Keep random', 'random')]:
            keep_idx = class_balanced_keep_indices(labels, scores_np, ratio, mode, SEED)
            subset = Subset(base_train_dataset, keep_idx.tolist())
            _, accs = train_and_record_curves(subset, val_dataset, EPOCHS_PRUNE)
            acc_curves[name].append(accs[-1])
    return acc_curves


def avg_pairwise_cosine(base_train_dataset: Dataset, indices: np.ndarray) -> float:
    if len(indices) < 2:
        return 0.0
    X = np.stack([base_train_dataset[i][0].view(-1).numpy() for i in indices]).astype(np.float32)
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    sim = X @ X.T
    triu = sim[np.triu_indices_from(sim, k=1)]
    return float(triu.mean())


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
        bottom = bbox.y1 - (r + 1) * cell_h - r * pad_y - 0.07
        subax = fig.add_axes([left, bottom, cell_w, cell_h])
        img, y = base_train_dataset[idx]
        subax.imshow(img.squeeze(0).numpy(), cmap='gray')
        subax.set_title(f'i={idx}\ny={y}\n{scores[idx]:.3f}', fontsize=8)
        subax.axis('off')


def make_summary_figure(mc_values, approx_values, raw_drop, raw_keep, topo_drop, topo_keep, base_train_dataset, raw_scores, topo_scores, save_path):
    pear = pearsonr(mc_values, approx_values)[0]
    spear = spearmanr(mc_values, approx_values)[0]
    raw_np = raw_scores.numpy()
    topo_np = topo_scores.numpy()
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.5])

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(mc_values, approx_values, s=40, alpha=0.85)
    mn = min(mc_values.min(), approx_values.min())
    mx = max(mc_values.max(), approx_values.max())
    if mx > mn:
        ax1.plot([mn, mx], [mn, mx], '--', color='gray', linewidth=1)
    ax1.set_title('Single-step MC Fidelity (raw Adam-aware)')
    ax1.set_xlabel('MC local Shapley')
    ax1.set_ylabel('Adam-aware approx')
    ax1.text(0.05, 0.95, f'Pearson = {pear:.3f}\nSpearman = {spear:.3f}\nN = {len(mc_values)}',
             transform=ax1.transAxes, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(np.array(DROP_RATIOS) * 100, raw_drop['Drop most helpful'], marker='o', label='Raw drop helpful')
    ax2.plot(np.array(DROP_RATIOS) * 100, topo_drop['Drop most helpful'], marker='o', label='Topo drop helpful')
    ax2.plot(np.array(DROP_RATIOS) * 100, raw_drop['Drop random'], marker='o', label='Raw drop random', linestyle='--')
    ax2.plot(np.array(DROP_RATIOS) * 100, topo_drop['Drop random'], marker='o', label='Topo drop random', linestyle='--')
    ax2.set_title('Drop-based pruning: raw vs topo-aware')
    ax2.set_xlabel('Removed ratio (%)')
    ax2.set_ylabel(f'Final val acc after {EPOCHS_PRUNE} epochs')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(np.array(KEEP_RATIOS) * 100, raw_keep['Keep most helpful'], marker='o', label='Raw keep helpful')
    ax3.plot(np.array(KEEP_RATIOS) * 100, topo_keep['Keep most helpful'], marker='o', label='Topo keep helpful')
    ax3.plot(np.array(KEEP_RATIOS) * 100, raw_keep['Keep random'], marker='o', label='Raw keep random', linestyle='--')
    ax3.plot(np.array(KEEP_RATIOS) * 100, topo_keep['Keep random'], marker='o', label='Topo keep random', linestyle='--')
    ax3.set_title('Keep-only: raw vs topo-aware')
    ax3.set_xlabel('Kept ratio (%)')
    ax3.set_ylabel(f'Final val acc after {EPOCHS_PRUNE} epochs')
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    raw_helpful = np.argsort(raw_np)[:TOPK_VIS]
    topo_helpful = np.argsort(topo_np)[:TOPK_VIS]
    txt = (
        'Topology regularization:\n'
        'score_topo(i) = raw(i) + alpha * sum_k A_ik raw(k) - lambda * sum_{j in batch} A_ij\n\n'
        f'K = {TOPO_K}, PCA dim = {TOPO_PCA_DIM}\n'
        f'alpha = {ALPHA_TOPO}, lambda = {LAMBDA_TOPO}\n\n'
        f'Raw helpful top-{TOPK_VIS} avg pairwise cosine = {avg_pairwise_cosine(base_train_dataset, raw_helpful):.3f}\n'
        f'Topo helpful top-{TOPK_VIS} avg pairwise cosine = {avg_pairwise_cosine(base_train_dataset, topo_helpful):.3f}'
    )
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    ax4.text(0.02, 0.98, txt, va='top', fontsize=10)

    ax5 = fig.add_subplot(gs[2, 0])
    plot_sample_grid(ax5, base_train_dataset, raw_helpful, 'Raw Adam-aware: most helpful samples', raw_np)

    ax6 = fig.add_subplot(gs[2, 1])
    plot_sample_grid(ax6, base_train_dataset, topo_helpful, 'Topo-aware Adam: most helpful samples', topo_np)

    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def main():
    set_seed(SEED)
    transform = transforms.ToTensor()
    base_train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    base_test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_dataset = IndexedDataset(base_train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_reference_loader = DataLoader(base_test_dataset, batch_size=1, shuffle=True)

    A_csr = build_knn_graph_from_raw_pixels(base_train_dataset, TOPO_K, TOPO_PCA_DIM)

    print('Step 1/4: computing raw Adam-aware and topo-aware scores ...')
    raw_scores, topo_scores, snapshot = compute_topo_adamaware_scores_and_capture_step(train_loader, val_reference_loader, base_train_dataset, A_csr, capture_step=0)

    print('Step 2/4: running single-step MC fidelity validation for raw Adam-aware component ...')
    mc_values, approx_values = monte_carlo_local_shapley_adam(snapshot, MC_BATCH_SIZE, MC_PERMUTATIONS)
    print(f'Pearson = {pearsonr(mc_values, approx_values)[0]:.4f}')
    print(f'Spearman = {spearmanr(mc_values, approx_values)[0]:.4f}')

    print('Step 3/4: running raw-score pruning / keep experiments ...')
    raw_drop = run_drop_curves(base_train_dataset, base_test_dataset, raw_scores)
    raw_keep = run_keep_curves(base_train_dataset, base_test_dataset, raw_scores)

    print('Step 4/4: running topo-score pruning / keep experiments ...')
    topo_drop = run_drop_curves(base_train_dataset, base_test_dataset, topo_scores)
    topo_keep = run_keep_curves(base_train_dataset, base_test_dataset, topo_scores)

    print(f'Saving summary figure to: {PLOT_PATH}')
    make_summary_figure(mc_values, approx_values, raw_drop, raw_keep, topo_drop, topo_keep, base_train_dataset, raw_scores, topo_scores, PLOT_PATH)
    print('Done.')


if __name__ == '__main__':
    main()
