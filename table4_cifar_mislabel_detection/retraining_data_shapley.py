"""Retraining-based Data Shapley adapter."""

from table4_common import load_method_scores


def score(context):
    values = load_method_scores(context, "data_shapley")
    return {"Retraining-based Data Shapley": values["Data Shapley"]}

