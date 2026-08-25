import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import scipy.sparse as sp

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)
np.random.seed(0)


# =========================================================
# 1) Dataset wrapper with true sample index
# =========================================================
class IndexedDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        x, y = self.base_dataset[idx]
        return x, y, idx


# =========================================================
# 2) Model
# =========================================================
class MLP_MNIST(nn.Module):
    def __init__(self):
        super(MLP_MNIST, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# =========================================================
# 3) Hook recorder for linear layers
# =========================================================
class LinearGhostRecorder:
    def __init__(self, module):
        self.x_in = None
        self.delta_out = None
        self.fwd_handle = module.register_forward_hook(self.forward_hook)
        self.bwd_handle = module.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, inp, out):
        self.x_in = inp[0].detach()

    def backward_hook(self, module, grad_input, grad_output):
        self.delta_out = grad_output[0].detach()

    def close(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()


# =========================================================
# 4) Utilities
# =========================================================
def zero_model_grads(model):
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

def safe_step_int(step_obj):
    if isinstance(step_obj, torch.Tensor):
        return int(step_obj.item())
    return int(step_obj)

def get_prev_adam_state(param, optimizer):
    state = optimizer.state[param]
    if len(state) == 0:
        exp_avg_prev = torch.zeros_like(param, device=param.device)
        exp_avg_sq_prev = torch.zeros_like(param, device=param.device)
        prev_step = 0
    else:
        exp_avg_prev = state["exp_avg"].detach().clone()
        exp_avg_sq_prev = state["exp_avg_sq"].detach().clone()
        prev_step = safe_step_int(state["step"])
    return exp_avg_prev, exp_avg_sq_prev, prev_step


def get_linearized_adam_coeffs(param, optimizer, beta1, beta2, eps):
    """
    Practical Linearized Ghost Approximation:
        u_i ≈ C1 ⊙ g_i - C2 ⊙ g_i^2 - C3 ⊙ g_i^3
    """
    exp_avg_prev, exp_avg_sq_prev, prev_step = get_prev_adam_state(param, optimizer)

    next_step = prev_step + 1
    bc1 = 1.0 - beta1 ** next_step
    bc2 = 1.0 - beta2 ** next_step

    if prev_step == 0:
        v_prev_hat = torch.zeros_like(param, device=param.device)
    else:
        prev_bc2 = 1.0 - beta2 ** prev_step
        v_prev_hat = exp_avg_sq_prev / prev_bc2

    A = torch.sqrt(v_prev_hat) + eps
    alpha = (1.0 - beta1) / bc1
    m_prev_term = beta1 * exp_avg_prev / bc1
    rho = (1.0 - beta2) / bc2

    C1 = alpha / A
    C2 = (rho * m_prev_term) / (2.0 * (A ** 3))
    C3 = (rho * alpha) / (2.0 * (A ** 3))

    return C1, C2, C3


def make_val_grad_from_last_sample(x_all, d_all):
    x_val = x_all[-1]
    d_val = d_all[-1]
    gW_val = d_val.unsqueeze(1) * x_val.unsqueeze(0)
    gB_val = d_val
    return x_val, d_val, gW_val, gB_val


def layer_linearized_ghost_scores(
    x_train, d_train,
    gW_val, gB_val,
    C1_w, C2_w, C3_w,
    C1_b, C2_b, C3_b
):
    """
    For one Linear layer:
      g_i = d_i ⊗ x_i
      u_i ≈ C1⊙g_i - C2⊙g_i^2 - C3⊙g_i^3
      score_i = <g_val, u_i>
    """
    # weight terms
    M1_w = C1_w * gW_val
    M2_w = C2_w * gW_val
    M3_w = C3_w * gW_val

    tmp1 = x_train @ M1_w.T
    score_w1 = (tmp1 * d_train).sum(dim=1)

    x2 = x_train ** 2
    d2 = d_train ** 2
    tmp2 = x2 @ M2_w.T
    score_w2 = (tmp2 * d2).sum(dim=1)

    x3 = x_train ** 3
    d3 = d_train ** 3
    tmp3 = x3 @ M3_w.T
    score_w3 = (tmp3 * d3).sum(dim=1)

    # bias terms
    vb1 = C1_b * gB_val
    vb2 = C2_b * gB_val
    vb3 = C3_b * gB_val

    score_b1 = (d_train * vb1.unsqueeze(0)).sum(dim=1)
    score_b2 = ((d_train ** 2) * vb2.unsqueeze(0)).sum(dim=1)
    score_b3 = ((d_train ** 3) * vb3.unsqueeze(0)).sum(dim=1)

    total = (score_w1 + score_b1) - (score_w2 + score_b2) - (score_w3 + score_b3)
    return total


# =========================================================
# 5) Topology graph construction
# =========================================================
def build_knn_graph_from_raw_pixels(base_dataset, k=10, pca_dim=64):
    """
    Build offline semantic/topology graph A from raw pixels.
    For your real project, replace raw pixels with better semantic features.
    Return: scipy CSR sparse matrix, row-normalized.
    """
    print("Building offline KNN graph ...")

    X = np.stack([base_dataset[i][0].view(-1).numpy() for i in range(len(base_dataset))]).astype(np.float32)

    # L2 normalize
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    # optional PCA
    if pca_dim is not None and pca_dim < X.shape[1]:
        X = PCA(n_components=pca_dim, random_state=0).fit_transform(X).astype(np.float32)
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    nbrs = NearestNeighbors(n_neighbors=k + 1, metric='cosine', n_jobs=-1)
    nbrs.fit(X)
    dist, ind = nbrs.kneighbors(X)

    # remove self neighbor
    rows = np.repeat(np.arange(X.shape[0]), k)
    cols = ind[:, 1:].reshape(-1)
    sims = 1.0 - dist[:, 1:].reshape(-1)
    sims = np.clip(sims, 0.0, None)

    A = sp.coo_matrix((sims, (rows, cols)), shape=(X.shape[0], X.shape[0]))

    # symmetrize
    A = 0.5 * (A + A.T)
    A.setdiag(0.0)
    A.eliminate_zeros()

    # row normalize
    row_sum = np.asarray(A.sum(axis=1)).ravel()
    row_sum[row_sum == 0] = 1.0
    A = sp.diags(1.0 / row_sum) @ A

    print("KNN graph built.")
    return A.tocsr()


def topology_regularize_batch_scores(
    batch_indices_np,
    raw_batch_scores_np,
    raw_score_buffer_np,
    A_csr,
    alpha=0.2,
    lambda_reg=0.05
):
    """
    Topology-Regularized score:
      S_tilde(i) = S_raw(i) + alpha * sum_k A_ik S_raw(k) - lambda * sum_{j in Batch} A_ij
    """
    # incorporate current batch raw scores into a temporary up-to-date buffer
    tmp_raw = raw_score_buffer_np.copy()
    tmp_raw[batch_indices_np] += raw_batch_scores_np

    # 1st-order topology diffusion
    diffusion_batch = A_csr[batch_indices_np].dot(tmp_raw)

    # 2nd-order in-batch crowding penalty
    crowding_batch = np.asarray(A_csr[batch_indices_np][:, batch_indices_np].sum(axis=1)).ravel()

    topo_batch_scores_np = raw_batch_scores_np + alpha * diffusion_batch - lambda_reg * crowding_batch
    return topo_batch_scores_np, diffusion_batch, crowding_batch


# =========================================================
# 6) Data
# =========================================================
transform = transforms.ToTensor()

base_train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, transform=transform, download=True
)
base_val_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, transform=transform, download=True
)

train_dataset = IndexedDataset(base_train_dataset)
val_dataset = IndexedDataset(base_val_dataset)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

val_iter = iter(val_loader)
val_img, val_label, _ = next(val_iter)
val_img = val_img.to(device)
val_label = val_label.to(device)


# =========================================================
# 7) Build topology graph once (offline)
# =========================================================
A_csr = build_knn_graph_from_raw_pixels(
    base_dataset=base_train_dataset,
    k=10,
    pca_dim=64
)

# hyperparameters for topology regularization
alpha_topo = 0.2
lambda_topo = 0.05


# =========================================================
# 8) Model / loss / optimizer
# =========================================================
model = MLP_MNIST().to(device)

criterion_mean = nn.CrossEntropyLoss()
criterion_none = nn.CrossEntropyLoss(reduction='none')

lr = 1e-3
beta1 = 0.9
beta2 = 0.999
eps = 1e-8

optimizer = optim.Adam(
    model.parameters(),
    lr=lr,
    betas=(beta1, beta2),
    eps=eps
)

# two buffers:
# 1) raw Adam-aware In-Run score
# 2) topology-regularized score
raw_score_buffer = torch.zeros(len(train_dataset), dtype=torch.float32)   # CPU
topo_score_buffer = torch.zeros(len(train_dataset), dtype=torch.float32)  # CPU

# attach recorders
rec_fc1 = LinearGhostRecorder(model.fc1)
rec_fc2 = LinearGhostRecorder(model.fc2)


# =========================================================
# 9) Training loop
# =========================================================
epochs = 1
global_step = 0

model.train()
for epoch in range(epochs):
    for batch_idx, (images, labels, indices) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        indices = indices.to(device)

        B = images.size(0)

        # -------------------------------------------
        # A) Adam-aware raw score via Linearized Ghost
        # -------------------------------------------
        combined_imgs = torch.cat([images, val_img], dim=0)
        combined_labels = torch.cat([labels, val_label], dim=0)

        zero_model_grads(model)
        logits_all = model(combined_imgs)

        train_losses = criterion_none(logits_all[:B], labels)
        val_loss = criterion_mean(logits_all[B:], val_label)

        combined_loss = train_losses.sum() + val_loss
        combined_loss.backward()

        x1_all, d1_all = rec_fc1.x_in, rec_fc1.delta_out
        x2_all, d2_all = rec_fc2.x_in, rec_fc2.delta_out

        x1_train, d1_train = x1_all[:B], d1_all[:B]
        x2_train, d2_train = x2_all[:B], d2_all[:B]

        _, _, gW1_val, gB1_val = make_val_grad_from_last_sample(x1_all, d1_all)
        _, _, gW2_val, gB2_val = make_val_grad_from_last_sample(x2_all, d2_all)

        C1_w1, C2_w1, C3_w1 = get_linearized_adam_coeffs(
            model.fc1.weight, optimizer, beta1, beta2, eps
        )
        C1_b1, C2_b1, C3_b1 = get_linearized_adam_coeffs(
            model.fc1.bias, optimizer, beta1, beta2, eps
        )
        C1_w2, C2_w2, C3_w2 = get_linearized_adam_coeffs(
            model.fc2.weight, optimizer, beta1, beta2, eps
        )
        C1_b2, C2_b2, C3_b2 = get_linearized_adam_coeffs(
            model.fc2.bias, optimizer, beta1, beta2, eps
        )

        score_fc1 = layer_linearized_ghost_scores(
            x_train=x1_train,
            d_train=d1_train,
            gW_val=gW1_val,
            gB_val=gB1_val,
            C1_w=C1_w1, C2_w=C2_w1, C3_w=C3_w1,
            C1_b=C1_b1, C2_b=C2_b1, C3_b=C3_b1
        )

        score_fc2 = layer_linearized_ghost_scores(
            x_train=x2_train,
            d_train=d2_train,
            gW_val=gW2_val,
            gB_val=gB2_val,
            C1_w=C1_w2, C2_w=C2_w2, C3_w=C3_w2,
            C1_b=C1_b2, C2_b=C2_b2, C3_b=C3_b2
        )

        raw_batch_scores = -lr * (score_fc1 + score_fc2)   # GPU tensor, shape [B]
        raw_batch_scores_np = raw_batch_scores.detach().cpu().numpy().astype(np.float32)

        # -------------------------------------------
        # B) Topology regularization
        # -------------------------------------------
        batch_indices_np = indices.detach().cpu().numpy()
        raw_score_buffer_np = raw_score_buffer.numpy()

        topo_batch_scores_np, diffusion_batch, crowding_batch = topology_regularize_batch_scores(
            batch_indices_np=batch_indices_np,
            raw_batch_scores_np=raw_batch_scores_np,
            raw_score_buffer_np=raw_score_buffer_np,
            A_csr=A_csr,
            alpha=alpha_topo,
            lambda_reg=lambda_topo
        )

        # accumulate
        raw_score_buffer[batch_indices_np] += torch.from_numpy(raw_batch_scores_np)
        topo_score_buffer[batch_indices_np] += torch.from_numpy(topo_batch_scores_np.astype(np.float32))

        # -------------------------------------------
        # C) Real Adam training step
        # -------------------------------------------
        zero_model_grads(model)
        logits = model(images)
        train_loss = criterion_mean(logits, labels)
        train_loss.backward()
        optimizer.step()

        global_step += 1

        if batch_idx % 100 == 0:
            print(
                f"Epoch {epoch}, Step {batch_idx}, "
                f"train_loss = {train_loss.item():.4f}, "
                f"val_loss = {val_loss.item():.4f}, "
                f"raw_mean = {raw_batch_scores_np.mean():.6f}, "
                f"diff_mean = {diffusion_batch.mean():.6f}, "
                f"crowd_mean = {crowding_batch.mean():.6f}"
            )

print("Topology-Regularized Adam-aware In-Run Shapley computed.")
print("raw_score_buffer shape :", raw_score_buffer.shape)
print("topo_score_buffer shape:", topo_score_buffer.shape)


# =========================================================
# 10) Ranking
# =========================================================
sorted_raw = torch.argsort(raw_score_buffer, descending=True)
sorted_topo = torch.argsort(topo_score_buffer, descending=True)

print("\n=== Highest RAW scores ===")
for rank in range(10):
    idx = sorted_raw[rank].item()
    val = raw_score_buffer[idx].item()
    print(f"Rank {rank+1:2d} | Sample Index: {idx:5d} | Raw Score: {val:.6f}")

print("\n=== Highest TOPO-REGULARIZED scores ===")
for rank in range(10):
    idx = sorted_topo[rank].item()
    val = topo_score_buffer[idx].item()
    print(f"Rank {rank+1:2d} | Sample Index: {idx:5d} | Topo Score: {val:.6f}")

print("\n=== Lowest TOPO-REGULARIZED scores ===")
for rank in range(10):
    idx = sorted_topo[-(rank+1)].item()
    val = topo_score_buffer[idx].item()
    print(f"Rank {rank+1:2d} | Sample Index: {idx:5d} | Topo Score: {val:.6f}")


# cleanup hooks
rec_fc1.close()
rec_fc2.close()