"""Exact KNN-Shapley recurrence on ResNet18 penultimate-layer features."""

import numpy as np
import torch
import torch.nn.functional as functional

from table5_common import DEVICE, RunContext, make_loader, standalone_main, train_model

METHOD_NAME = "KNN-Shapley"


@torch.no_grad()
def _features(model, dataset, context):
    loader = make_loader(dataset, context.config, shuffle=False, seed=0)
    model.eval()
    features, labels = [], []
    original_fc = model.fc
    model.fc = torch.nn.Identity()
    for images, target in loader:
        features.append(functional.normalize(model(images.to(DEVICE)), dim=1).cpu())
        labels.append(target.long())
    model.fc = original_fc
    return torch.cat(features), torch.cat(labels)


def score(context: RunContext) -> np.ndarray:
    model = train_model(context.train_dataset, context.config, context.seed + 501)
    train_x, train_y = _features(model, context.train_dataset, context)
    val_x, val_y = _features(model, context.valuation_dataset, context)
    train_x, train_y = train_x.to(DEVICE), train_y.to(DEVICE)
    n = train_x.size(0)
    k = min(context.config.knn_k, n)
    rank = torch.arange(1, n, device=DEVICE, dtype=torch.float32)
    coefficient = torch.minimum(rank, torch.tensor(float(k), device=DEVICE)) / (k * rank)
    values = torch.zeros(n, device=DEVICE)
    chunk_size = max(1, min(128, context.config.batch_size * 2))
    for start in range(0, len(val_x), chunk_size):
        query = val_x[start : start + chunk_size].to(DEVICE)
        query_labels = val_y[start : start + chunk_size].to(DEVICE)
        order = torch.cdist(query, train_x).argsort(dim=1)
        matches = (train_y[order] == query_labels[:, None]).float()
        delta = (matches[:, :-1] - matches[:, 1:]) * coefficient[None]
        sorted_values = torch.empty_like(matches)
        sorted_values[:, -1] = matches[:, -1] / n
        sorted_values[:, :-1] = sorted_values[:, -1:] + torch.flip(
            torch.cumsum(torch.flip(delta, dims=[1]), dim=1), dims=[1]
        )
        values.scatter_add_(0, order.flatten(), sorted_values.flatten())
    return (values / len(val_x)).cpu().numpy()


if __name__ == "__main__":
    standalone_main(METHOD_NAME, score)

