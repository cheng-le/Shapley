"""LESS-style checkpoint gradient-similarity baseline."""

import numpy as np

from table5_common import RunContext, make_projection, projected_checkpoint_score, standalone_main, train_model

METHOD_NAME = "LESS"


def score(context: RunContext) -> np.ndarray:
    cfg = context.config
    projection = make_projection(cfg, context.seed + 401)
    requested = {min(epoch, cfg.epochs) for epoch in (10, 20, 30, 40, 50)}
    values = []

    def callback(epoch, model):
        if epoch in requested:
            values.append(projected_checkpoint_score(model, context, projection, mode="cosine"))

    train_model(context.train_dataset, cfg, context.seed + 402, callback=callback)
    return np.mean(values, axis=0)


if __name__ == "__main__":
    standalone_main(METHOD_NAME, score)

