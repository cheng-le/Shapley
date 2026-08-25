"""TRAK-style projected-gradient baseline with 1/5/15/25 checkpoints."""

from collections import defaultdict

import numpy as np

from table5_common import RunContext, make_projection, projected_checkpoint_score, standalone_main, train_model

METHOD_NAME = "TRAK"
CHECKPOINT_COUNTS = (1, 5, 15, 25)


def score(context: RunContext) -> dict[str, np.ndarray]:
    cfg = context.config
    projection = make_projection(cfg, context.seed + 301)
    first_epoch = max(1, cfg.epochs - max(CHECKPOINT_COUNTS) + 1)
    checkpoint_scores = []

    def callback(epoch, model):
        if epoch >= first_epoch:
            checkpoint_scores.append(projected_checkpoint_score(model, context, projection, mode="trak"))

    train_model(context.train_dataset, cfg, context.seed + 302, callback=callback)
    results = {}
    for count in CHECKPOINT_COUNTS:
        actual = min(count, len(checkpoint_scores))
        results[f"TRAK ({count} cpt{'s' if count != 1 else ''})"] = np.mean(checkpoint_scores[-actual:], axis=0)
    return results


if __name__ == "__main__":
    standalone_main(METHOD_NAME, score)

