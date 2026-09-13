"""Second-order In-Run Data Shapley with a last-layer empirical-Fisher HVP."""

import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from table5_common import DEVICE, RunContext, _last_layer_batch, make_loader, make_model, standalone_main

METHOD_NAME = "2nd Order In-Run Data Shapley"


def score(context: RunContext) -> np.ndarray:
    cfg = context.config
    model = make_model(cfg, context.seed + 701)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.lr_step, gamma=cfg.lr_gamma)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    values = torch.zeros(len(context.train_dataset), dtype=torch.float64)
    indexed = torch.utils.data.TensorDataset(torch.arange(len(context.train_dataset)))
    order_loader = make_loader(indexed, cfg, shuffle=True, seed=context.seed + 702)
    val_loader = make_loader(context.valuation_dataset, cfg, shuffle=True, seed=context.seed + 703)

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
                lr_now = optimizer.param_groups[0]["lr"]
                steps = -lr_now / len(indices) * train_grads
                total_step = steps.sum(dim=0)
                hvp_total = val_grads.T @ (val_grads @ total_step) / val_grads.size(0)
                first = lr_now / len(indices) * (train_grads @ val_mean)
                curvature = 0.5 * (steps @ hvp_total)
                local = first - curvature
            values[indices] += local.double().cpu()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
    return values.numpy()


if __name__ == "__main__":
    standalone_main(METHOD_NAME, score)

