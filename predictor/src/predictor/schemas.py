"""Defines the data structures for the prediction service."""

from typing import Any

from msgspec import Struct


class PredictionRequest(Struct):
    """Represents a prediction request."""

    instances: list[list[Any] | dict[str, Any]]
    parameters: dict[str, Any] | None = {}


class PredictionResponse(Struct):
    """Represents a prediction response."""

    predictions: list[list[float] | dict[str, Any]]
