"""Defines the data structures for the prediction service."""

from typing import Any

from msgspec import Struct


class PredictionRequest(Struct):
    """Represents a prediction request."""

    instances: list[list[Any] | dict[str, Any]]
    parameters: dict[str, Any] | None = {}


class AutoMLComponents(Struct):
    """Holds the components of the result."""

    classes: list[str]
    scores: list[float]


class PredictionResponse(Struct):
    """Represents a prediction response

    A response either conforms to the AutoML response format or contains a list of key-value pairs,
    class label to score.
    """

    predictions: list[AutoMLComponents | dict[str, float]]
