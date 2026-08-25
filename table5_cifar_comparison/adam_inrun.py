"""Adam-aware first/second-order In-Run baselines for the extended Table 5."""

import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from table5_common import DEVICE, RunContext, _last_layer_batch, make_loader, make_model, standalone_main

METHOD_NAME = "Adam-aware In-Run"


def _step_number(state: dict) -> int:
    value = state.get("step", 0)
    return int(value.item()) if isinstance(value, torch.Tensor) else int(value)


def _adam_directions(model, optimizer, sample_grads: torch.Tensor) -> torch.Tensor:
    group = optimizer.param_groups[0]
    weight, bias = model.fc.weight, model.fc.bias
    weight_state, bias_state = optimizer.state[weight], optimizer.state[bias]
    if weight_state:
        previous_mean = torch.cat([weight_state["exp_avg"].flatten(), bias_state["exp_avg"].flatten()])
        previous_square = torch.cat([weight_state["exp_avg_sq"].flatten(), bias_state["exp_avg_sq"].flatten()])
        previous_step = _step_number(weight_state)
    else:
        previous_mean = torch.zeros(sample_grads.size(1), device=sample_grads.device)
        previous_square = torch.zeros_like(previous_mean)
        previous_step = 0
    parameters = torch.cat([weight.detach().flatten(), bias.detach().flatten()])
    gradients = sample_grads + group["weight_decay"] * parameters[None]
    beta1, beta2 = group["betas"]
    mean = beta1 * previous_mean[None] + (1.0 - beta1) * gradients
    square = beta2 * previous_square[None] + (1.0 - beta2) * gradients.square()
    step = previous_step + 1
    mean = mean / (1.0 - beta1**step)
    square = square / (1.0 - beta2**step)
    return mean / (square.sqrt() + group["eps"])


def score(context: RunContext) -> dict[str, np.ndarray]:
    cfg = context.config
    model = make_model(cfg, context.seed + 901)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.lr_step, gamma=cfg.lr_gamma)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    first_values = torch.zeros(len(context.train_dataset), dtype=torch.float64)
    second_values = torch.zeros_like(first_values)
    indexed = torch.utils.data.TensorDataset(torch.arange(len(context.train_dataset)))
    order_loader = make_loader(indexed, cfg, shuffle=True, seed=context.seed + 902)
    val_loader = make_loader(context.valuation_dataset, cfg, shuffle=True, seed=context.seed + 903)

    for _epoch in range(cfg.epochs):
        val_iter = itertools.cycle(val_loader)
        for (indices,) in order_loader:
            images = torch.stack([context.train_dataset[int(i)][0] for i in indices]).to(DEVICE)
            labels = torch.tensor([context.train_dataset[int(i)][1] for i in indices], device=DEVICE)
            val_images, val_labels = next(val_iter)
            val_images, val_labels = val_images.to(DEVICE), val_labels.to(DEVICE)
            model.eval()
            with torch.no_grad():
                train_grads = _last_layer_batch(model, images, labels, cfg.label_smoothing)
                val_grads = _last_layer_batch(model, val_images, val_labels, cfg.label_smoothing)
                val_mean = val_grads.mean(dim=0)
                directions = _adam_directions(model, optimizer, train_grads)
                steps = -optimizer.param_groups[0]["lr"] * directions
                first = -(steps @ val_mean)
                total_step = steps.sum(dim=0)
                hvp_total = val_grads.T @ (val_grads @ total_step) / val_grads.size(0)
                second = first - 0.5 * (steps @ hvp_total)
            first_values[indices] += first.double().cpu()
            second_values[indices] += second.double().cpu()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_fn(model(images), labels).backward()
            optimizer.step()
        scheduler.step()
    return {
        "Adam-aware 1st Order In-Run": first_values.numpy(),
        "Adam-aware 2nd Order In-Run": second_values.numpy(),
    }


if __name__ == "__main__":
    standalone_main(METHOD_NAME, score)
