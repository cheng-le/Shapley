"""Shared protocol for the CIFAR10 mislabeled-data detection experiment."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
TABLE5_DIR = ROOT / "table5_cifar_comparison"
TABLE5_RESULTS = TABLE5_DIR / "results"
for path in (ROOT, TABLE5_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from table5_common import (  # noqa: E402
    Table5Config,
    build_run_context,
    config_fingerprint,
    load_or_compute_scores,
)


EXPECTED_FINGERPRINT = "16a6f42d74e8"


def paper_aligned_config() -> Table5Config:
    """Return the exact Appendix E.6 configuration used by the Table 5 run."""
    config = Table5Config()
    fingerprint = config_fingerprint(config)
    if fingerprint != EXPECTED_FINGERPRINT:
        raise RuntimeError(
            f"Table 4 protocol drifted from the completed Table 5 run: "
            f"expected {EXPECTED_FINGERPRINT}, got {fingerprint}"
        )
    return config


def build_context(run_index: int):
    context = build_run_context(paper_aligned_config(), run_index)
    if len(context.train_dataset) != 1000:
        raise RuntimeError(f"Expected 1000 training points, got {len(context.train_dataset)}")
    if int(context.corrupted_mask.sum()) != 100:
        raise RuntimeError(f"Expected 100 mislabeled points, got {int(context.corrupted_mask.sum())}")
    return context


def load_method_scores(context, module_name: str) -> dict[str, np.ndarray]:
    """Load the Table 5 score cache, computing it only when the cache is absent."""
    module = importlib.import_module(module_name)
    cache_version = getattr(module, "CACHE_VERSION", "")
    cache_key = module.METHOD_NAME
    if cache_version:
        cache_key = f"{cache_key} {cache_version}"
    variants = load_or_compute_scores(context, cache_key, module.score)
    normalized = {}
    for name, values in variants.items():
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (len(context.train_dataset),):
            raise ValueError(f"{name} returned shape {array.shape}; expected (1000,)")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains non-finite scores")
        normalized[name] = array
    return normalized


def mislabeled_auroc(corrupted_mask: np.ndarray, data_values: np.ndarray) -> float:
    """Treat mislabeled points as positives and low data value as anomalous."""
    return float(roc_auc_score(corrupted_mask.astype(np.uint8), -data_values))

