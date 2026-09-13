"""One-run first/second-order attribution along a real AdamW trajectory."""

from __future__ import annotations

import numpy as np
import torch

from pile_main_100k.common import encode, example_losses, gradient_vector, stratified_sample_rows, trainable


def trajectory_attribution(net, tok, rows, validation, device, seed, batch_size=8, learning_rate=6e-4):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rows))
    parameters = trainable(net)
    dimension = sum(parameter.numel() for parameter in parameters)
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1)
    gradients = np.zeros((len(rows), dimension), dtype=np.float16)
    directions = np.zeros_like(gradients)
    losses_out = np.zeros(len(rows), dtype=np.float32)
    score_arrays = {name: np.zeros(len(rows), dtype=np.float32) for name in (
        "1st Order In-Run Data Shapley", "2nd Order In-Run Data Shapley",
        "Adam-aware 1st Order In-Run", "Adam-aware 2nd Order In-Run",
    )}
    val_rows = stratified_sample_rows(validation, 16, seed=2027)

    for batch_number, start in enumerate(range(0, len(order), batch_size), start=1):
        indices = order[start : start + batch_size]
        batch = encode(tok, [rows[int(index)]["text"] for index in indices], device)
        val_batch = encode(tok, [row["text"] for row in val_rows], device)
        net.train()
        val_loss = example_losses(net, val_batch).mean()
        val_parts = torch.autograd.grad(val_loss, parameters, create_graph=True)
        query = torch.cat([part.reshape(-1) for part in val_parts])
        sample_losses = example_losses(net, batch)
        local = []
        for position, sample_loss in enumerate(sample_losses):
            local.append(gradient_vector(sample_loss, parameters, retain_graph=position + 1 < len(sample_losses)).detach())
        grad = torch.stack(local)

        moments, variances, step = [], [], 1
        for parameter in parameters:
            state = optimizer.state.get(parameter, {})
            moments.append(state.get("exp_avg", torch.zeros_like(parameter)).reshape(-1))
            variances.append(state.get("exp_avg_sq", torch.zeros_like(parameter)).reshape(-1))
            if "step" in state:
                step = int(state["step"].item()) + 1
        moment, variance = torch.cat(moments), torch.cat(variances)
        next_moment = 0.9 * moment[None] + 0.1 * grad
        next_variance = 0.95 * variance[None] + 0.05 * grad.square()
        direction = (next_moment / (1.0 - 0.9**step)) / ((next_variance / (1.0 - 0.95**step)).sqrt() + 1e-8)

        scale = learning_rate / len(indices)
        first = scale * (grad * query.detach()).sum(1)
        adam_first = scale * (direction * query.detach()).sum(1)
        total_sgd_step = -scale * grad.sum(0)
        total_adam_step = -scale * direction.sum(0)
        hvp_sgd_parts = torch.autograd.grad((query * total_sgd_step).sum(), parameters, retain_graph=True)
        hvp_adam_parts = torch.autograd.grad((query * total_adam_step).sum(), parameters)
        hvp_sgd = torch.cat([part.reshape(-1) for part in hvp_sgd_parts]).detach()
        hvp_adam = torch.cat([part.reshape(-1) for part in hvp_adam_parts]).detach()
        curvature_sgd = -0.5 * ((-scale * grad) * hvp_sgd).sum(1)
        curvature_adam = -0.5 * ((-scale * direction) * hvp_adam).sum(1)
        for base, curvature in ((first, curvature_sgd), (adam_first, curvature_adam)):
            base_scale, curve_scale = base.std(), curvature.std()
            if curve_scale > 0.25 * base_scale and curve_scale > 1e-12:
                curvature.mul_(0.25 * base_scale / curve_scale)

        gradients[indices] = grad.cpu().numpy().astype(np.float16)
        directions[indices] = direction.cpu().numpy().astype(np.float16)
        losses_out[indices] = sample_losses.detach().cpu().numpy()
        score_arrays["1st Order In-Run Data Shapley"][indices] = first.detach().cpu().numpy()
        score_arrays["2nd Order In-Run Data Shapley"][indices] = (first + curvature_sgd).detach().cpu().numpy()
        score_arrays["Adam-aware 1st Order In-Run"][indices] = adam_first.detach().cpu().numpy()
        score_arrays["Adam-aware 2nd Order In-Run"][indices] = (adam_first + curvature_adam).detach().cpu().numpy()

        optimizer.zero_grad(set_to_none=True)
        update_loss = example_losses(net, batch).mean()
        update_loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        if batch_number % 100 == 0:
            print(f"trajectory rows={min(start + len(indices), len(rows)):,}/{len(rows):,}", flush=True)
    return gradients, directions, losses_out, score_arrays
