import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)
np.random.seed(0)


# -----------------------------
# 1) Dataset wrapper with index
# -----------------------------
class IndexedDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        x, y = self.base_dataset[idx]
        return x, y, idx


# -----------------------------
# 2) Model
# -----------------------------
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


# -----------------------------
# 3) Hook recorder for linear layers
# -----------------------------
class LinearGhostRecorder:
    def __init__(self, module):
        self.x_in = None
        self.delta_out = None
        self.fwd_handle = module.register_forward_hook(self.forward_hook)
        self.bwd_handle = module.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, inp, out):
        # inp[0]: [B, in_features]
        self.x_in = inp[0].detach()

    def backward_hook(self, module, grad_input, grad_output):
        # grad_output[0]: dL / d(linear_output), shape [B, out_features]
        self.delta_out = grad_output[0].detach()

    def close(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()


# -----------------------------
# 4) Helpers
# -----------------------------
def zero_model_grads(model):
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()


def safe_step_int(step_obj):
    if isinstance(step_obj, torch.Tensor):
        return int(step_obj.item())
    return int(step_obj)


def get_prev_adam_state(param, optimizer):
    """
    Return previous Adam state before current optimizer.step():
      exp_avg_prev, exp_avg_sq_prev, prev_step
    """
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
    Practical Linearized Ghost Approximation coefficients.

    We approximate the Adam update direction for one sample as:
        u_i ≈ C1 ⊙ g_i - C2 ⊙ g_i^2 - C3 ⊙ g_i^3

    where the coefficients come from a first-order Taylor expansion
    of the Adam denominator around the previous second-moment state.
    """
    exp_avg_prev, exp_avg_sq_prev, prev_step = get_prev_adam_state(param, optimizer)

    next_step = prev_step + 1
    bc1 = 1.0 - beta1 ** next_step
    bc2 = 1.0 - beta2 ** next_step

    # previous bias-corrected second moment
    if prev_step == 0:
        v_prev_hat = torch.zeros_like(param, device=param.device)
    else:
        prev_bc2 = 1.0 - beta2 ** prev_step
        v_prev_hat = exp_avg_sq_prev / prev_bc2

    A = torch.sqrt(v_prev_hat) + eps

    # first-order coefficient for current gradient
    alpha = (1.0 - beta1) / bc1

    # approximate historical moment contribution
    m_prev_term = beta1 * exp_avg_prev / bc1

    # variance increment coefficient
    rho = (1.0 - beta2) / bc2

    C1 = alpha / A
    C2 = (rho * m_prev_term) / (2.0 * (A ** 3))
    C3 = (rho * alpha) / (2.0 * (A ** 3))

    return C1, C2, C3


def make_val_grad_from_last_sample(x_all, d_all):
    """
    Given combined batch activations/errors:
      x_all: [B+1, in_dim]
      d_all: [B+1, out_dim]
    last row is validation sample
    """
    x_val = x_all[-1]           # [in_dim]
    d_val = d_all[-1]           # [out_dim]

    gW_val = d_val.unsqueeze(1) * x_val.unsqueeze(0)  # [out_dim, in_dim]
    gB_val = d_val                                     # [out_dim]
    return x_val, d_val, gW_val, gB_val


def layer_linearized_ghost_scores(
    x_train, d_train,
    gW_val, gB_val,
    C1_w, C2_w, C3_w,
    C1_b, C2_b, C3_b
):
    """
    Compute per-sample score contributions for one Linear layer.

    For sample i, weight gradient is:
        gW_i = d_i ⊗ x_i
    bias gradient is:
        gB_i = d_i

    We approximate update direction as:
        u_i ≈ C1 ⊙ g_i - C2 ⊙ g_i^2 - C3 ⊙ g_i^3

    and compute:
        <g_val, u_i>
    """
    # ---------- Weight terms ----------
    M1_w = C1_w * gW_val
    M2_w = C2_w * gW_val
    M3_w = C3_w * gW_val

    # term1: d_i^T M1 x_i
    tmp1 = x_train @ M1_w.T                  # [B, out_dim]
    score_w1 = (tmp1 * d_train).sum(dim=1)   # [B]

    # term2: (d_i^2)^T M2 (x_i^2)
    x2 = x_train ** 2
    d2 = d_train ** 2
    tmp2 = x2 @ M2_w.T
    score_w2 = (tmp2 * d2).sum(dim=1)

    # term3: (d_i^3)^T M3 (x_i^3)
    x3 = x_train ** 3
    d3 = d_train ** 3
    tmp3 = x3 @ M3_w.T
    score_w3 = (tmp3 * d3).sum(dim=1)

    # ---------- Bias terms ----------
    vb1 = C1_b * gB_val
    vb2 = C2_b * gB_val
    vb3 = C3_b * gB_val

    score_b1 = (d_train * vb1.unsqueeze(0)).sum(dim=1)
    score_b2 = ((d_train ** 2) * vb2.unsqueeze(0)).sum(dim=1)
    score_b3 = ((d_train ** 3) * vb3.unsqueeze(0)).sum(dim=1)

    total = (score_w1 + score_b1) - (score_w2 + score_b2) - (score_w3 + score_b3)
    return total


# -----------------------------
# 5) Data
# -----------------------------
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


# -----------------------------
# 6) Model / loss / optimizer
# -----------------------------
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

data_shapley_scores = torch.zeros(len(train_dataset), device=device)

# attach recorders
rec_fc1 = LinearGhostRecorder(model.fc1)
rec_fc2 = LinearGhostRecorder(model.fc2)


# -----------------------------
# 7) Training loop with Linearized Ghost Approximation
# -----------------------------
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
        # A) Attribution pass: one combined backward
        # -------------------------------------------
        combined_imgs = torch.cat([images, val_img], dim=0)
        combined_labels = torch.cat([labels, val_label], dim=0)

        zero_model_grads(model)
        logits_all = model(combined_imgs)

        train_losses = criterion_none(logits_all[:B], labels)   # [B]
        val_loss = criterion_mean(logits_all[B:], val_label)    # scalar

        # sum(train_losses) gives per-sample deltas in grad_output rows
        combined_loss = train_losses.sum() + val_loss
        combined_loss.backward()

        # recorded activations / backprops for combined batch
        x1_all, d1_all = rec_fc1.x_in, rec_fc1.delta_out
        x2_all, d2_all = rec_fc2.x_in, rec_fc2.delta_out

        # split train / val
        x1_train, d1_train = x1_all[:B], d1_all[:B]
        x2_train, d2_train = x2_all[:B], d2_all[:B]

        _, _, gW1_val, gB1_val = make_val_grad_from_last_sample(x1_all, d1_all)
        _, _, gW2_val, gB2_val = make_val_grad_from_last_sample(x2_all, d2_all)

        # Adam coefficients from previous optimizer state
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

        # Layer-wise ghost scores
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

        # total batch attribution
        batch_scores = -lr * (score_fc1 + score_fc2)

        # accumulate to dataset-level buffer
        data_shapley_scores[indices] += batch_scores.detach()

        # -------------------------------------------
        # B) Real training step (Adam)
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
                f"val_loss = {val_loss.item():.4f}"
            )

print("Linearized Adam-Ghost In-Run Data Shapley computed.")
print("data_shapley_scores shape:", data_shapley_scores.shape)

sorted_idx = torch.argsort(data_shapley_scores, descending=True)

print("\nHighest Shapley scores")
for rank in range(10):
    idx = sorted_idx[rank].item()
    val = data_shapley_scores[idx].item()
    print(f"Rank {rank+1:2d} | Sample Index: {idx:5d} | Score: {val:.6f}")

print("\nLowest Shapley scores")
for rank in range(10):
    idx = sorted_idx[-(rank+1)].item()
    val = data_shapley_scores[idx].item()
    print(f"Rank {rank+1:2d} | Sample Index: {idx:5d} | Score: {val:.6f}")

# cleanup hooks
rec_fc1.close()
rec_fc2.close()