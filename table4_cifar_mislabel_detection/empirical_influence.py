"""Empirical Influence Function adapter."""

from table4_common import load_method_scores


def score(context):
    return load_method_scores(context, "empirical_influence")

