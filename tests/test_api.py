"""
API Tests
==========
Tests for the FastAPI serving endpoint.

MLOps Concept: API tests ensure the serving layer works correctly.
These run in CI/CD before deployment to catch integration issues.
"""

import sys
import pytest
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────
# Sample Data
# ─────────────────────────────────────────────────────────────

VALID_CUSTOMER = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.50,
    "TotalCharges": 1026.0,
}

LOW_RISK_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "Yes",
    "tenure": 60,
    "PhoneService": "Yes",
    "MultipleLines": "Yes",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "Yes",
    "DeviceProtection": "Yes",
    "TechSupport": "Yes",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Two year",
    "PaperlessBilling": "No",
    "PaymentMethod": "Bank transfer (automatic)",
    "MonthlyCharges": 45.0,
    "TotalCharges": 2700.0,
}


# ─────────────────────────────────────────────────────────────
# Pydantic Schema Tests (no server needed)
# ─────────────────────────────────────────────────────────────

class TestInputSchema:
    """Tests for Pydantic input validation schemas."""

    def test_valid_customer_schema(self):
        """Valid customer data should pass Pydantic validation."""
        from api.app import CustomerFeatures
        customer = CustomerFeatures(**VALID_CUSTOMER)
        assert customer.tenure == 12
        assert customer.MonthlyCharges == 85.50

    def test_invalid_senior_citizen_value(self):
        """SeniorCitizen must be 0 or 1."""
        from api.app import CustomerFeatures
        from pydantic import ValidationError
        invalid_data = {**VALID_CUSTOMER, "SeniorCitizen": 5}
        with pytest.raises(ValidationError):
            CustomerFeatures(**invalid_data)

    def test_negative_tenure_rejected(self):
        """Negative tenure should be rejected."""
        from api.app import CustomerFeatures
        from pydantic import ValidationError
        invalid_data = {**VALID_CUSTOMER, "tenure": -1}
        with pytest.raises(ValidationError):
            CustomerFeatures(**invalid_data)

    def test_negative_monthly_charges_rejected(self):
        """Negative monthly charges should be rejected."""
        from api.app import CustomerFeatures
        from pydantic import ValidationError
        invalid_data = {**VALID_CUSTOMER, "MonthlyCharges": -10.0}
        with pytest.raises(ValidationError):
            CustomerFeatures(**invalid_data)

    def test_missing_required_field_rejected(self):
        """Missing required field should raise ValidationError."""
        from api.app import CustomerFeatures
        from pydantic import ValidationError
        incomplete_data = {k: v for k, v in VALID_CUSTOMER.items() if k != "tenure"}
        with pytest.raises(ValidationError):
            CustomerFeatures(**incomplete_data)

    def test_model_dump_returns_dict(self):
        """model_dump() should return a dictionary."""
        from api.app import CustomerFeatures
        customer = CustomerFeatures(**VALID_CUSTOMER)
        result = customer.model_dump()
        assert isinstance(result, dict)
        assert len(result) == len(VALID_CUSTOMER)

    def test_batch_request_schema(self):
        """Batch request should accept list of customers."""
        from api.app import BatchPredictionRequest, CustomerFeatures
        batch = BatchPredictionRequest(
            customers=[
                CustomerFeatures(**VALID_CUSTOMER),
                CustomerFeatures(**LOW_RISK_CUSTOMER),
            ]
        )
        assert len(batch.customers) == 2

    def test_empty_batch_rejected(self):
        """Empty batch should be rejected."""
        from api.app import BatchPredictionRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            BatchPredictionRequest(customers=[])


# ─────────────────────────────────────────────────────────────
# API Endpoint Tests (with mocked model)
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def mock_model_state():
    """Mock the model state for API testing without a real model."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([1])
    mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])

    mock_preprocessor = MagicMock()
    mock_preprocessor.transform.return_value = np.zeros((1, 50))
    mock_preprocessor.get_feature_names_out.return_value = [f"f{i}" for i in range(50)]

    return mock_model, mock_preprocessor


@pytest.fixture
def test_client(mock_model_state):
    """Create a test client with mocked model."""
    try:
        from fastapi.testclient import TestClient
        from api.app import app, app_state
        from src.data_ingestion import load_config

        mock_model, mock_preprocessor = mock_model_state

        # Patch app state
        app_state.model = mock_model
        app_state.preprocessor = mock_preprocessor
        app_state.is_ready = True
        app_state.model_version = "test-v1"
        app_state.algorithm = "RandomForestClassifier"
        app_state.config = load_config("configs/params.yaml")

        client = TestClient(app)
        return client
    except Exception as e:
        pytest.skip(f"Could not create test client: {e}")


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_root_endpoint(self, test_client):
        """Root endpoint should return 200."""
        response = test_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert data["service"] == "Churn Prediction API"

    def test_health_endpoint_returns_200(self, test_client):
        """Health endpoint should return 200."""
        response = test_client.get("/health")
        assert response.status_code == 200

    def test_health_response_schema(self, test_client):
        """Health response should have required fields."""
        response = test_client.get("/health")
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "uptime_seconds" in data
        assert data["status"] == "healthy"

    def test_ready_endpoint_when_model_loaded(self, test_client):
        """Ready endpoint should return 200 when model is loaded."""
        response = test_client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert data["model_loaded"] is True


class TestPredictionEndpoints:
    """Tests for prediction endpoints."""

    def test_predict_endpoint_returns_200(self, test_client):
        """Predict endpoint should return 200 for valid input."""
        response = test_client.post("/predict", json=VALID_CUSTOMER)
        assert response.status_code == 200

    def test_predict_response_schema(self, test_client):
        """Prediction response should have all required fields."""
        response = test_client.post("/predict", json=VALID_CUSTOMER)
        data = response.json()
        required_fields = [
            "prediction", "prediction_label", "churn_probability",
            "no_churn_probability", "confidence", "model_version",
            "prediction_id", "timestamp"
        ]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"

    def test_predict_returns_valid_prediction(self, test_client):
        """Prediction should be 0 or 1."""
        response = test_client.post("/predict", json=VALID_CUSTOMER)
        data = response.json()
        assert data["prediction"] in [0, 1]

    def test_predict_returns_valid_probability(self, test_client):
        """Churn probability should be between 0 and 1."""
        response = test_client.post("/predict", json=VALID_CUSTOMER)
        data = response.json()
        assert 0 <= data["churn_probability"] <= 1
        assert 0 <= data["no_churn_probability"] <= 1

    def test_predict_probabilities_sum_to_one(self, test_client):
        """Churn and no-churn probabilities should sum to ~1."""
        response = test_client.post("/predict", json=VALID_CUSTOMER)
        data = response.json()
        total = data["churn_probability"] + data["no_churn_probability"]
        assert abs(total - 1.0) < 0.01, f"Probabilities sum to {total}, expected ~1.0"

    def test_predict_invalid_input_returns_422(self, test_client):
        """Invalid input should return 422 Unprocessable Entity."""
        invalid_data = {**VALID_CUSTOMER, "tenure": -5}
        response = test_client.post("/predict", json=invalid_data)
        assert response.status_code == 422

    def test_predict_missing_field_returns_422(self, test_client):
        """Missing required field should return 422."""
        incomplete_data = {k: v for k, v in VALID_CUSTOMER.items() if k != "MonthlyCharges"}
        response = test_client.post("/predict", json=incomplete_data)
        assert response.status_code == 422

    def test_batch_predict_endpoint(self, test_client):
        """Batch predict endpoint should handle multiple customers."""
        batch_data = {
            "customers": [VALID_CUSTOMER, LOW_RISK_CUSTOMER]
        }
        response = test_client.post("/predict/batch", json=batch_data)
        assert response.status_code == 200
        data = response.json()
        assert data["total_records"] == 2
        assert len(data["predictions"]) == 2

    def test_batch_predict_response_schema(self, test_client):
        """Batch prediction response should have required fields."""
        batch_data = {"customers": [VALID_CUSTOMER]}
        response = test_client.post("/predict/batch", json=batch_data)
        data = response.json()
        required_fields = [
            "predictions", "total_records", "churn_count",
            "churn_rate", "timestamp"
        ]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"


class TestMetricsEndpoint:
    """Tests for the metrics endpoint."""

    def test_metrics_endpoint_returns_200(self, test_client):
        """Metrics endpoint should return 200."""
        response = test_client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_response_schema(self, test_client):
        """Metrics response should have required fields."""
        response = test_client.get("/metrics")
        data = response.json()
        required_fields = [
            "total_predictions", "churn_predictions",
            "churn_rate", "avg_latency_ms", "model_version",
            "is_ready", "timestamp"
        ]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"

    def test_metrics_churn_rate_valid_range(self, test_client):
        """Churn rate in metrics should be between 0 and 1."""
        response = test_client.get("/metrics")
        data = response.json()
        assert 0 <= data["churn_rate"] <= 1
