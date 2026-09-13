"""Retraining-based Data Shapley via coalition utilities and KernelSHAP regression."""

import numpy as np
from sklearn.linear_model import Ridge

from table5_common import RunContext, retraining_ensemble, standalone_main

METHOD_NAME = "Data Shapley"


def score(context: RunContext) -> np.ndarray:
    masks, utilities = retraining_ensemble(context)
    sizes = masks.sum(axis=1).astype(np.float64)
    n = masks.shape[1]
    weights = 1.0 / np.maximum(sizes * (n - sizes), 1.0)
    weights /= weights.mean()
    model = Ridge(alpha=context.config.datamodel_ridge, fit_intercept=True)
    model.fit(masks.astype(np.float64), utilities, sample_weight=weights)
    return model.coef_.astype(np.float64)


if __name__ == "__main__":
    standalone_main(METHOD_NAME, score)

