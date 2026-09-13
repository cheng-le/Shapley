"""Adam-aware first/second-order In-Run adapters."""

from table4_common import load_method_scores


def score(context):
    return load_method_scores(context, "adam_inrun")

