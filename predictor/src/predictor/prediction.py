"""Prediction service for the application."""

from typing import Any

import pandas as pd
from autogluon.tabular import TabularPredictor


from predictor.utils import parse_instances_to_dataframe
from predictor.schemas import Parameters, PredictionResponse


def create_prediction(
    model: TabularPredictor,
    instances: list[dict[str, Any] | list[Any]],
    parameters: Parameters = Parameters(),
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

    predictions = model.predict_proba(
        df_to_predict, as_pandas=parameters.as_object
    )

    if isinstance(predictions, pd.DataFrame):
        response_data = predictions.to_dict(orient="records")
    else:
        response_data = predictions.tolist()

    return response_data  # type: ignore[return-value]
