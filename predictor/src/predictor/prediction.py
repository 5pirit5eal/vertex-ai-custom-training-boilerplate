"""Prediction service for the application."""

import logging
from typing import Any

import pandas as pd
from autogluon.tabular import TabularPredictor

from predictor.utils import parse_instances_to_dataframe


def create_prediction(
    model: TabularPredictor,
    instances: list[dict[str, Any] | list[Any]],
    parameters: dict[str, Any] | None = None,
) -> list[dict[str, Any] | list[float]]:
    """Generates predictions for a given list of instances.

    Args:
        model: The trained AutoGluon model.
        instances: A list of instances to predict.
        parameters: A dictionary of parameters for prediction.

    Returns:
        list[dict[str, Any] | list[float]]: The predictions.

    Raises:
        ValueError: If the problem type is not supported.
    """
    if parameters is None:
        parameters = {}
    df_to_predict, is_list = parse_instances_to_dataframe(instances, model)

    if model.problem_type not in ["binary", "multiclass"]:
        raise ValueError("Unsupported problem type")

    as_object = parameters.get("as_object")
    if as_object is None:
        as_pandas = not is_list
    else:
        as_pandas = as_object

    predictions = model.predict_proba(df_to_predict, as_pandas=as_pandas)

    if isinstance(predictions, pd.DataFrame):
        response_data = predictions.to_dict(orient="records")
    else:
        response_data = predictions.tolist()

    return response_data  # type: ignore[return-value]
