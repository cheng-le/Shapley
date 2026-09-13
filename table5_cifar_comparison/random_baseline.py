"""Random data-selection baseline from Table 5."""

import numpy as np

from table5_common import RunContext, standalone_main

METHOD_NAME = "Random"


def score(context: RunContext) -> np.ndarray:
    return np.random.default_rng(context.seed + 101).random(len(context.train_dataset))


if __name__ == "__main__":
    standalone_main(METHOD_NAME, score)

