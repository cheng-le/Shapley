"""Empirical Influence Function estimated from random-subset retraining runs."""

import numpy as np

from table5_common import RunContext, retraining_ensemble, standalone_main

METHOD_NAME = "Empirical Influence Function"


def score(context: RunContext) -> np.ndarray:
    masks, utilities = retraining_ensemble(context)
    included_count = masks.sum(axis=0).clip(min=1)
    excluded_count = (~masks).sum(axis=0).clip(min=1)
    included_mean = masks.T @ utilities / included_count
    excluded_mean = (~masks).T @ utilities / excluded_count
    return (included_mean - excluded_mean).astype(np.float64)


if __name__ == "__main__":
    standalone_main(METHOD_NAME, score)

