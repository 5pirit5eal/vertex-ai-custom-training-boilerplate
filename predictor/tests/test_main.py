import pytest
from litestar.testing import TestClient
from predictor.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_ping(client):
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.text == "pong"


def test_predict_success(monkeypatch, client):
    # Patch predictor.predict to return a fixed value
    from predictor import main

    monkeypatch.setattr(main.predictor, "predict", lambda df: [42] * len(df))
    data = {
        "instances": [
            {"feature1": 1, "feature2": 2},
            {"feature1": 3, "feature2": 4},
        ]
    }
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    result = response.json()
    assert "predictions" in result
    assert result["predictions"] == [42, 42]


def test_predict_error(monkeypatch, client):
    from predictor import main

    monkeypatch.setattr(
        main.predictor, "predict", lambda df: 1 / 0
    )  # Force error
    data = {"instances": [{"feature1": 1}]}
    response = client.post("/predict", json=data)
    assert response.status_code == 500
    result = response.json()
    assert "error" in result
