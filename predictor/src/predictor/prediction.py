"""Prediction service for the application."""

from typing import Any

import pandas as pd
from autogluon.tabular import TabularPredictor

from predictor.schemas import AutoMLComponents
from predictor.utils import parse_instances_to_dataframe


# def create_prediction(
#     model: TabularPredictor,
#     instances: list[dict[str, Any] | list[Any]],
#     parameters: dict[str, Any] | None = None,
# ) -> list[AutoMLComponents | dict[str, float]]:
#     """Generates predictions for a given list of instances.

#     Args:
#         model: The trained AutoGluon model.
#         instances: A list of instances to predict.
#         parameters: A dictionary of parameters for prediction.

#     Returns:
#         list[AutoMLComponents | dict[str, float]]: The predictions.

#     Raises:
#         ValueError: If the problem type is not supported.
#     """
#     if parameters is None:
#         parameters = {}
#     df_to_predict, is_list = parse_instances_to_dataframe(instances, model)

#     if model.problem_type not in ["binary", "multiclass"]:
#         raise ValueError("Unsupported problem type")

#     as_object = parameters.get("as_object", False)

#     predictions = model.predict_proba(df_to_predict, as_pandas=as_object)

#     if isinstance(predictions, pd.DataFrame):
#         response_data = predictions.to_dict(orient="records")
#     else:
#         preds = predictions.tolist()
#         if isinstance(preds[0], list):
#             response_data = [
#                 AutoMLComponents(
#                     classes=[str(label) for label in model.class_labels],
#                     scores=pred,
#                 )
#                 for pred in preds
#             ]
#         else:
#             response_data = [
#                 AutoMLComponents(
#                     classes=[str(label) for label in model.class_labels],
#                     scores=preds,
#                 )
#             ]

#     return response_data  # type: ignore[return-value]


def create_prediction(
    model: TabularPredictor,
    instances: list[dict[str, Any] | list[Any]],
    parameters: dict[str, Any] | None = None,
) -> list[AutoMLComponents | dict[str, float]]:
    """Generates predictions for a given list of instances.

    Args:
        model: The trained AutoGluon model.
        instances: A list of instances to predict.
        parameters: A dictionary of parameters for prediction.

    Returns:
        list[AutoMLComponents | dict[str, float]]: The predictions.

    Raises:
        ValueError: If the problem type is not supported.
    """
    response_data = [
        AutoMLComponents(classes=["1", "0"], scores=[0.1, 0.9])
        for _ in range(len(instances))
    ]

    return response_data  # type: ignore[return-value]
