"""Tests for the prediction service."""

from typing import Any
from unittest.mock import MagicMock, ANY

import pandas as pd
import pytest
from autogluon.tabular import TabularPredictor

from predictor.prediction import create_prediction
from predictor.schemas import Parameters


@pytest.fixture
def mock_predictor() -> MagicMock:
    """Returns a mock TabularPredictor."""
    predictor = MagicMock(spec=TabularPredictor)
    predictor.feature_metadata_in.to_dict.return_value = {
        "feat1": "int",
        "feat2": "float",
    }
    predictor.problem_type = "binary"
    return predictor


def test_create_prediction_with_dict_instances(
    mock_predictor: MagicMock,
) -> None:
    """Tests the create_prediction with dictionary instances."""
    instances: list[dict[str, Any] | list[Any]] = [{"feat1": 1, "feat2": 2.0}]
    mock_predictor.predict_proba.return_value = pd.DataFrame(
        [[0.1, 0.9]], columns=["class1", "class2"]
    )

    result = create_prediction(mock_predictor, instances)

    assert result == [{"class1": 0.1, "class2": 0.9}]


def test_create_prediction_with_list_instances(
    mock_predictor: MagicMock,
) -> None:
    """Tests the create_prediction with list instances."""
    instances: list[dict[str, Any] | list[Any]] = [[1, 2.0]]
    mock_predictor.predict_proba.return_value = pd.DataFrame(
        [[0.1, 0.9]], columns=["class1", "class2"]
    ).values

    result = create_prediction(mock_predictor, instances)

    assert result == [[0.1, 0.9]]


def test_create_prediction_with_as_object_true(
    mock_predictor: MagicMock,
) -> None:
    """Tests the create_prediction with as_object=True."""
    instances: list[dict[str, Any] | list[Any]] = [[1, 2.0]]
    parameters = Parameters(as_object=True)
    mock_predictor.predict_proba.return_value = pd.DataFrame(
        [[0.1, 0.9]], columns=["class1", "class2"]
    )

    result = create_prediction(mock_predictor, instances, parameters)

    assert result == [{"class1": 0.1, "class2": 0.9}]
    mock_predictor.predict_proba.assert_called_once_with(ANY, as_pandas=True)


def test_create_prediction_with_as_object_false(
    mock_predictor: MagicMock,
) -> None:
    """Tests the create_prediction with as_object=False."""
    instances: list[dict[str, Any] | list[Any]] = [{"feat1": 1, "feat2": 2.0}]
    parameters = Parameters(as_object=False)
    mock_predictor.predict_proba.return_value = pd.DataFrame(
        [[0.1, 0.9]], columns=["class1", "class2"]
    ).values

    result = create_prediction(mock_predictor, instances, parameters)

    assert result == [[0.1, 0.9]]


def test_create_prediction_unsupported_problem_type(
    mock_predictor: MagicMock,
) -> None:
    """Tests the create_prediction with an unsupported problem type."""
    mock_predictor.problem_type = "regression"
    instances: list[dict[str, Any] | list[Any]] = [{"feat1": 1, "feat2": 2.0}]

    with pytest.raises(ValueError, match="Unsupported problem type"):
        create_prediction(mock_predictor, instances)
