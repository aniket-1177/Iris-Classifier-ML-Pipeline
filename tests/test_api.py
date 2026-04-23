"""
Integration tests for the FastAPI application.

Uses TestClient — no running server needed.
The Predictor is mocked so tests don't require trained model artifacts.

Run with:
    pytest tests/test_api.py -v
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Mock the predictor before importing the app
MOCK_RESULT = {
    "predicted_class": "setosa",
    "confidence": 0.95,
    "class_probabilities": {
        "setosa": 0.95,
        "versicolor": 0.03,
        "virginica": 0.02,
    },
}

MOCK_CLASSES = ["setosa", "versicolor", "virginica"]


@pytest.fixture(scope="module")
def client():
    mock_predictor = MagicMock()
    mock_predictor.predict.return_value = MOCK_RESULT
    mock_predictor.predict_batch.return_value = [MOCK_RESULT, MOCK_RESULT]
    mock_predictor.model_classes = MOCK_CLASSES

    # with patch("src.inference.predictor.get_predictor", return_value=mock_predictor):
    with patch("api.routers.predict.get_predictor", return_value=mock_predictor):
        from api.main import create_app

        app = create_app()
        with TestClient(app) as c:
            yield c


# ── Health & Root ─────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_schema(self, client):
        body = client.get("/health").json()
        assert "status" in body
        assert "model_loaded" in body
        assert "version" in body

    def test_root_returns_message(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()


# ── Single Prediction ─────────────────────────────────────────────────────────

VALID_PAYLOAD = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2,
}


class TestPredictEndpoint:
    def test_valid_request_200(self, client):
        response = client.post("/predict/", json=VALID_PAYLOAD)
        assert response.status_code == 200

    def test_response_schema(self, client):
        body = client.post("/predict/", json=VALID_PAYLOAD).json()
        assert "predicted_class" in body
        assert "confidence" in body
        assert "class_probabilities" in body

    def test_predicted_class_is_string(self, client):
        body = client.post("/predict/", json=VALID_PAYLOAD).json()
        assert isinstance(body["predicted_class"], str)

    def test_missing_field_422(self, client):
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "sepal_length"}
        response = client.post("/predict/", json=payload)
        assert response.status_code == 422

    def test_negative_value_422(self, client):
        payload = {**VALID_PAYLOAD, "sepal_length": -1.0}
        response = client.post("/predict/", json=payload)
        assert response.status_code == 422

    def test_value_too_large_422(self, client):
        payload = {**VALID_PAYLOAD, "petal_length": 999.0}
        response = client.post("/predict/", json=payload)
        assert response.status_code == 422


# ── Batch Prediction ──────────────────────────────────────────────────────────

class TestBatchPredictEndpoint:
    def test_batch_200(self, client):
        payload = {"samples": [VALID_PAYLOAD, VALID_PAYLOAD]}
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 200

    def test_batch_total_count(self, client):
        payload = {"samples": [VALID_PAYLOAD, VALID_PAYLOAD]}
        body = client.post("/predict/batch", json=payload).json()
        assert body["total"] == 2
        assert len(body["predictions"]) == 2

    def test_empty_batch_422(self, client):
        response = client.post("/predict/batch", json={"samples": []})
        assert response.status_code == 422
