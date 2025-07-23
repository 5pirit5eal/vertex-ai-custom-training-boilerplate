"""Defines the data structures for the prediction service."""

from typing import Any

from msgspec import Struct


class Parameters(Struct, frozen=True):
    """Represents additional parameters for prediction requests."""

    as_object: bool = False


class PredictionRequest(Struct):
    """Represents a prediction request."""

    instances: list[list[Any] | dict[str, Any]]
    parameters: Parameters = Parameters()


class PredictionResponse(Struct):
    """Represents a prediction response."""

    predictions: list[list[float] | dict[str, Any]]
