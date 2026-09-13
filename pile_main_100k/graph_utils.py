"""Memory-bounded exact cosine KNN for medium-scale Pile experiments."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def gpu_chunked_knn(features: np.ndarray, device: torch.device, k: int = 10, chunk_size: int = 512):
    vectors = F.normalize(torch.from_numpy(features.astype(np.float32)).to(device), dim=1)
    all_indices, all_weights = [], []
    for start in range(0, len(vectors), chunk_size):
        similarities = vectors[start : start + chunk_size] @ vectors.T
        row = torch.arange(len(similarities), device=device)
        similarities[row, start + row] = -torch.inf
        values, indices = similarities.topk(k, dim=1)
        values = values.clamp_min(0.0)
        values /= values.sum(1, keepdim=True).clamp_min(1e-8)
        all_indices.append(indices.cpu().numpy().astype(np.int32))
        all_weights.append(values.cpu().numpy().astype(np.float32))
        if (start // chunk_size + 1) % 20 == 0:
            print(f"knn rows={min(start + chunk_size, len(vectors)):,}/{len(vectors):,}", flush=True)
    del vectors
    return np.concatenate(all_indices), np.concatenate(all_weights)
