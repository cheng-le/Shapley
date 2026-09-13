"""Influence-function baseline using a final-layer empirical-Fisher IHVP."""

import numpy as np
import torch

from table5_common import RunContext, last_layer_gradients, standalone_main, train_model

METHOD_NAME = "Influence function"


def _conjugate_gradient(matvec, rhs: torch.Tensor, steps: int, tolerance: float = 1e-8) -> torch.Tensor:
    solution = torch.zeros_like(rhs)
    residual = rhs.clone()
    direction = residual.clone()
    residual_sq = torch.dot(residual, residual)
    for _ in range(steps):
        product = matvec(direction)
        alpha = residual_sq / (torch.dot(direction, product) + 1e-12)
        solution += alpha * direction
        residual -= alpha * product
        next_sq = torch.dot(residual, residual)
        if float(next_sq.sqrt()) < tolerance:
            break
        direction = residual + (next_sq / (residual_sq + 1e-12)) * direction
        residual_sq = next_sq
    return solution


def score(context: RunContext) -> np.ndarray:
    cfg = context.config
    model = train_model(context.train_dataset, cfg, context.seed + 201)
    train_grads = last_layer_gradients(model, context.train_dataset, cfg).float()
    val_grad = last_layer_gradients(model, context.valuation_dataset, cfg, reduce_mean=True).float()
    scale = float(train_grads.square().sum(dim=1).mean().item()) + 1e-12

    def fisher_vector_product(vector: torch.Tensor) -> torch.Tensor:
        return train_grads.T @ (train_grads @ vector) / train_grads.size(0) + cfg.influence_damping * scale * vector

    ihvp = _conjugate_gradient(fisher_vector_product, val_grad, cfg.influence_cg_steps)
    values = train_grads @ ihvp
    return values.numpy()


if __name__ == "__main__":
    standalone_main(METHOD_NAME, score)

