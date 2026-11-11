"""
Unit & Integration Test Suite for Final Project ML API
Tests all endpoints, schema validation, edge cases, error handling, and model inference.
"""

import sys
import os
import pytest
from fastapi.testclient import TestClient

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.main import app

@pytest.fixture(scope="module")
def client():
    """Initializes TestClient context triggering lifespan handlers."""
    with TestClient(app) as test_client:
        yield test_client

def test_root_endpoint(client):
    """GET / must return API metadata and 200 OK."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "version" in data
    assert "/health" in data["endpoints"]

def test_healthcheck(client):
    """GET /health must return healthy status and verify model is loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "uptime_seconds" in data

def test_metrics_endpoint(client):
    """GET /metrics must report operational telemetry."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "predictions_served" in data
    assert "uptime_seconds" in data

def test_predict_setosa(client):
    """POST /predict for typical Setosa features."""
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["class_name"] == "setosa"
    assert data["class_id"] == 0
    assert data["probabilities"]["setosa"] > 0.80

def test_predict_versicolor(client):
    """POST /predict for typical Versicolor features."""
    payload = {
        "sepal_length": 6.0,
        "sepal_width": 2.7,
        "petal_length": 4.0,
        "petal_width": 1.2
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["class_name"] == "versicolor"
    assert data["class_id"] == 1
    assert data["probabilities"]["versicolor"] > 0.50

def test_predict_virginica(client):
    """POST /predict for typical Virginica features."""
    payload = {
        "sepal_length": 6.9,
        "sepal_width": 3.1,
        "petal_length": 5.4,
        "petal_width": 2.1
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["class_name"] == "virginica"
    assert data["class_id"] == 2
    assert data["probabilities"]["virginica"] > 0.60

def test_predict_batch(client):
    """POST /predict/batch for multiple samples simultaneously."""
    batch_payload = {
        "samples": [
            {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
            {"sepal_length": 6.9, "sepal_width": 3.1, "petal_length": 5.4, "petal_width": 2.1}
        ]
    }
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert len(data["predictions"]) == 2
    assert data["predictions"][0]["class_name"] == "setosa"
    assert data["predictions"][1]["class_name"] == "virginica"

def test_validation_negative_values(client):
    """Input validation: negative dimensions must be rejected with 422."""
    invalid_payload = {
        "sepal_length": -5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422

def test_validation_missing_field(client):
    """Input validation: missing required field must be rejected with 422."""
    incomplete_payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4
    }
    response = client.post("/predict", json=incomplete_payload)
    assert response.status_code == 422

def test_validation_type_mismatch(client):
    """Input validation: non-numeric value must be rejected with 422."""
    type_err_payload = {
        "sepal_length": "invalid_number",
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post("/predict", json=type_err_payload)
    assert response.status_code == 422
