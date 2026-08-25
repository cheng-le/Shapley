"""Datamodels baseline fit to utilities from random training subsets."""

import numpy as np
from sklearn.linear_model import Ridge

from table5_common import RunContext, retraining_ensemble, standalone_main

METHOD_NAME = "Datamodels"


def score(context: RunContext) -> np.ndarray:
    masks, utilities = retraining_ensemble(context)
    model = Ridge(alpha=context.config.datamodel_ridge, fit_intercept=True)
    model.fit(masks.astype(np.float64), utilities)
    return model.coef_.astype(np.float64)


if __name__ == "__main__":
    standalone_main(METHOD_NAME, score)

