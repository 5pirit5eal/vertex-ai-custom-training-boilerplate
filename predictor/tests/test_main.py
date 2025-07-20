"""Tests for the main API."""

from unittest.mock import MagicMock, patch

import pytest
from litestar.status_codes import (
    HTTP_200_OK,
    HTTP_201_CREATED,
    HTTP_500_INTERNAL_SERVER_ERROR,
    HTTP_503_SERVICE_UNAVAILABLE,
)
from litestar.testing import TestClient

from predictor.main import app


@pytest.fixture
def test_client() -> TestClient:
    """Returns a TestClient for the application."""
    return TestClient(app)


def test_health_check_model_not_ready(test_client: TestClient) -> None:
    """Tests the health check endpoint when the model is not ready."""
    app.state.is_model_ready = False
    response = test_client.get("/health")
    assert response.status_code == HTTP_503_SERVICE_UNAVAILABLE
    assert response.text == "Model not ready"


def test_health_check_model_ready(test_client: TestClient) -> None:
    """Tests the health check endpoint when the model is ready."""
    app.state.is_model_ready = True
    response = test_client.get("/health")
    assert response.status_code == HTTP_200_OK
    assert response.text == "Model is ready"


@patch("predictor.main.create_prediction")
def test_predict_endpoint(
    mock_create_prediction: MagicMock, test_client: TestClient
) -> None:
    """Tests the predict endpoint."""
    app.state.is_model_ready = True
    app.state.predictor = MagicMock()
    mock_create_prediction.return_value = [[0.1, 0.9]]

    response = test_client.post("/predict", json={"instances": [[1, 2.0]]})
    assert response.status_code == HTTP_201_CREATED
    assert response.json() == {"predictions": [[0.1, 0.9]]}


def test_predict_endpoint_model_not_ready(test_client: TestClient) -> None:
    """Tests the predict endpoint when the model is not ready."""
    app.state.is_model_ready = False
    response = test_client.post("/predict", json={"instances": [[1, 2.0]]})
    assert response.status_code == HTTP_503_SERVICE_UNAVAILABLE


@patch("predictor.main.create_prediction")
def test_predict_endpoint_with_exception(
    mock_create_prediction: MagicMock, test_client: TestClient
) -> None:
    """Tests the predict endpoint when an exception is raised."""
    app.state.is_model_ready = True
    app.state.predictor = MagicMock()
    mock_create_prediction.side_effect = Exception("Test Exception")

    response = test_client.post("/predict", json={"instances": [[1, 2.0]]})
    assert response.status_code == HTTP_500_INTERNAL_SERVER_ERROR
