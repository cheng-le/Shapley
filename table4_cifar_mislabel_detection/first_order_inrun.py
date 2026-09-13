"""First-order In-Run Data Shapley adapter."""

from table4_common import load_method_scores


def score(context):
    return load_method_scores(context, "first_order_inrun")

