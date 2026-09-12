"""
Model Tests
============
Tests for model training, evaluation, and quality gates.

MLOps Concept: Model tests ensure the model meets minimum
performance standards before deployment. These run in CI/CD
and block deployment if thresholds are not met.

Test Categories:
- Model training tests
- Performance threshold tests (quality gates)
- Model behavior tests (sanity checks)
- Inference tests
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_ingestion import generate_synthetic_churn_data, load_config
from src.feature_engineering import preprocess_raw_data, engineer_features
from src.train import get_model, compute_metrics, check_model_quality_gate


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def config():
    return load_config("configs/params.yaml")


@pytest.fixture(scope="module")
def training_data(config):
    """Generate and preprocess training data."""
    df = generate_synthetic_churn_data(n_samples=1000, random_state=42)
    processed = preprocess_raw_data(df, config)
    X, y, preprocessor, feature_names = engineer_features(
        processed, config, fit=True, save_preprocessor=False
    )
    return X, y, preprocessor, feature_names


@pytest.fixture(scope="module")
def trained_model(config, training_data):
    """Train a model for testing."""
    X, y, _, _ = training_data
    model, algorithm, params = get_model(config)
    model.fit(X, y)
    return model


# ─────────────────────────────────────────────────────────────
# Model Instantiation Tests
# ─────────────────────────────────────────────────────────────

class TestModelInstantiation:
    """Tests for model factory function."""

    def test_random_forest_instantiation(self, config):
        """Random Forest model should instantiate correctly."""
        config_copy = dict(config)
        config_copy["model"] = dict(config["model"])
        config_copy["model"]["algorithm"] = "random_forest"
        model, algorithm, params = get_model(config_copy)
        assert isinstance(model, RandomForestClassifier)
        assert algorithm == "random_forest"

    def test_logistic_regression_instantiation(self, config):
        """Logistic Regression model should instantiate correctly."""
        config_copy = dict(config)
        config_copy["model"] = dict(config["model"])
        config_copy["model"]["algorithm"] = "logistic_regression"
        model, algorithm, params = get_model(config_copy)
        assert isinstance(model, LogisticRegression)
        assert algorithm == "logistic_regression"

    def test_invalid_algorithm_raises_error(self, config):
        """Invalid algorithm name should raise ValueError."""
        config_copy = dict(config)
        config_copy["model"] = dict(config["model"])
        config_copy["model"]["algorithm"] = "invalid_algo"
        with pytest.raises(ValueError, match="Unknown algorithm"):
            get_model(config_copy)

    def test_model_has_fit_method(self, config):
        """Model should have a fit method."""
        model, _, _ = get_model(config)
        assert hasattr(model, "fit"), "Model should have fit method"

    def test_model_has_predict_method(self, config):
        """Model should have a predict method."""
        model, _, _ = get_model(config)
        assert hasattr(model, "predict"), "Model should have predict method"

    def test_model_has_predict_proba_method(self, config):
        """Model should have predict_proba method for probability scores."""
        model, _, _ = get_model(config)
        assert hasattr(model, "predict_proba"), \
            "Model should have predict_proba method"


# ─────────────────────────────────────────────────────────────
# Model Training Tests
# ─────────────────────────────────────────────────────────────

class TestModelTraining:
    """Tests for model training correctness."""

    def test_model_trains_without_error(self, training_data, config):
        """Model should train without raising exceptions."""
        X, y, _, _ = training_data
        model, _, _ = get_model(config)
        model.fit(X, y)  # Should not raise

    def test_trained_model_makes_predictions(self, trained_model, training_data):
        """Trained model should make predictions."""
        X, y, _, _ = training_data
        predictions = trained_model.predict(X[:10])
        assert len(predictions) == 10, "Should return 10 predictions"

    def test_predictions_are_binary(self, trained_model, training_data):
        """Predictions should be 0 or 1."""
        X, y, _, _ = training_data
        predictions = trained_model.predict(X)
        valid_values = {0, 1}
        assert set(np.unique(predictions)).issubset(valid_values), \
            f"Predictions should be 0 or 1, got: {set(np.unique(predictions))}"

    def test_probabilities_sum_to_one(self, trained_model, training_data):
        """Prediction probabilities should sum to 1 for each sample."""
        X, y, _, _ = training_data
        probabilities = trained_model.predict_proba(X[:20])
        row_sums = probabilities.sum(axis=1)
        np.testing.assert_allclose(
            row_sums, np.ones(20), atol=1e-6,
            err_msg="Probabilities should sum to 1"
        )

    def test_probabilities_in_valid_range(self, trained_model, training_data):
        """Probabilities should be between 0 and 1."""
        X, y, _, _ = training_data
        probabilities = trained_model.predict_proba(X)
        assert (probabilities >= 0).all(), "Probabilities should be >= 0"
        assert (probabilities <= 1).all(), "Probabilities should be <= 1"

    def test_model_predicts_both_classes(self, trained_model, training_data):
        """Model should predict both churn and no-churn classes."""
        X, y, _, _ = training_data
        predictions = trained_model.predict(X)
        assert 0 in predictions, "Model should predict class 0 (No Churn)"
        assert 1 in predictions, "Model should predict class 1 (Churn)"


# ─────────────────────────────────────────────────────────────
# Metrics Tests
# ─────────────────────────────────────────────────────────────

class TestMetrics:
    """Tests for metric computation."""

    def test_compute_metrics_returns_all_metrics(self, trained_model, training_data):
        """compute_metrics should return all required metrics."""
        X, y, _, _ = training_data
        y_pred = trained_model.predict(X)
        y_prob = trained_model.predict_proba(X)[:, 1]
        metrics = compute_metrics(y, y_pred, y_prob)

        required_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        for metric in required_metrics:
            assert metric in metrics, f"Missing metric: {metric}"

    def test_metrics_in_valid_range(self, trained_model, training_data):
        """All metrics should be between 0 and 1."""
        X, y, _, _ = training_data
        y_pred = trained_model.predict(X)
        y_prob = trained_model.predict_proba(X)[:, 1]
        metrics = compute_metrics(y, y_pred, y_prob)

        for metric_name, value in metrics.items():
            assert 0 <= value <= 1, \
                f"Metric '{metric_name}' = {value} is outside [0, 1]"

    def test_perfect_predictions_give_perfect_metrics(self):
        """Perfect predictions should give metrics of 1.0."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.1, 0.9, 0.1, 0.9])
        metrics = compute_metrics(y_true, y_pred, y_prob)

        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0
        assert metrics["roc_auc"] == 1.0


# ─────────────────────────────────────────────────────────────
# Quality Gate Tests
# ─────────────────────────────────────────────────────────────

class TestQualityGate:
    """Tests for the model quality gate."""

    def test_good_model_passes_quality_gate(self, config):
        """Model with metrics above thresholds should pass quality gate."""
        good_metrics = {
            "accuracy": 0.85,
            "precision": 0.80,
            "recall": 0.75,
            "f1": 0.77,
            "roc_auc": 0.88,
        }
        result = check_model_quality_gate(good_metrics, config)
        assert result is True, "Good model should pass quality gate"

    def test_bad_model_fails_quality_gate(self, config):
        """Model with metrics below thresholds should fail quality gate."""
        bad_metrics = {
            "accuracy": 0.60,  # Below 0.75 threshold
            "precision": 0.55,
            "recall": 0.50,
            "f1": 0.52,        # Below 0.65 threshold
            "roc_auc": 0.65,   # Below 0.80 threshold
        }
        result = check_model_quality_gate(bad_metrics, config)
        assert result is False, "Bad model should fail quality gate"

    def test_borderline_model_at_threshold(self, config):
        """Model exactly at threshold should pass."""
        borderline_metrics = {
            "accuracy": 0.75,   # Exactly at threshold
            "precision": 0.70,
            "recall": 0.65,
            "f1": 0.65,         # Exactly at threshold
            "roc_auc": 0.80,    # Exactly at threshold
        }
        result = check_model_quality_gate(borderline_metrics, config)
        assert result is True, "Model at threshold should pass"

    def test_trained_model_passes_quality_gate(self, trained_model, training_data, config):
        """
        The trained model should pass the quality gate.
        This is the key CI/CD test - if this fails, deployment is blocked.
        """
        X, y, _, _ = training_data
        y_pred = trained_model.predict(X)
        y_prob = trained_model.predict_proba(X)[:, 1]
        metrics = compute_metrics(y, y_pred, y_prob)

        # Note: We use training metrics here for speed
        # In production CI/CD, use held-out test set metrics
        result = check_model_quality_gate(metrics, config)
        assert result is True, \
            f"Trained model failed quality gate. Metrics: {metrics}"


# ─────────────────────────────────────────────────────────────
# Model Behavior / Sanity Tests
# ─────────────────────────────────────────────────────────────

class TestModelBehavior:
    """
    Sanity tests for model behavior.
    
    MLOps Concept: Beyond accuracy metrics, we test that the model
    behaves sensibly - e.g., high-risk customers should have higher
    churn probability than low-risk customers.
    """

    def test_high_risk_customer_has_higher_churn_prob(
        self, trained_model, training_data, config
    ):
        """
        A month-to-month customer with short tenure should have
        higher churn probability than a 2-year contract customer.
        """
        X, y, preprocessor, _ = training_data

        # Create high-risk customer profile
        high_risk = {
            "gender": "Male", "SeniorCitizen": 1, "Partner": "No",
            "Dependents": "No", "tenure": 1, "PhoneService": "Yes",
            "MultipleLines": "No", "InternetService": "Fiber optic",
            "OnlineSecurity": "No", "OnlineBackup": "No",
            "DeviceProtection": "No", "TechSupport": "No",
            "StreamingTV": "Yes", "StreamingMovies": "Yes",
            "Contract": "Month-to-month", "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 95.0, "TotalCharges": 95.0,
            "Churn": 0
        }

        # Create low-risk customer profile
        low_risk = {
            "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
            "Dependents": "Yes", "tenure": 60, "PhoneService": "Yes",
            "MultipleLines": "Yes", "InternetService": "DSL",
            "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
            "DeviceProtection": "Yes", "TechSupport": "Yes",
            "StreamingTV": "No", "StreamingMovies": "No",
            "Contract": "Two year", "PaperlessBilling": "No",
            "PaymentMethod": "Bank transfer (automatic)",
            "MonthlyCharges": 45.0, "TotalCharges": 2700.0,
            "Churn": 0
        }

        from src.feature_engineering import preprocess_raw_data

        def get_churn_prob(customer_dict):
            df = pd.DataFrame([customer_dict])
            df = preprocess_raw_data(df, config)
            X_feat, _, _, _ = engineer_features(
                df, config, preprocessor=preprocessor,
                fit=False, save_preprocessor=False
            )
            return trained_model.predict_proba(X_feat)[0][1]

        high_risk_prob = get_churn_prob(high_risk)
        low_risk_prob = get_churn_prob(low_risk)

        assert high_risk_prob > low_risk_prob, (
            f"High-risk customer ({high_risk_prob:.3f}) should have higher "
            f"churn probability than low-risk ({low_risk_prob:.3f})"
        )

    def test_model_output_shape(self, trained_model, training_data):
        """Model output shape should match input batch size."""
        X, y, _, _ = training_data
        batch_sizes = [1, 10, 100]
        for batch_size in batch_sizes:
            predictions = trained_model.predict(X[:batch_size])
            assert len(predictions) == batch_size, \
                f"Expected {batch_size} predictions, got {len(predictions)}"

    def test_model_is_deterministic(self, trained_model, training_data):
        """Same input should always produce same output."""
        X, y, _, _ = training_data
        pred1 = trained_model.predict(X[:20])
        pred2 = trained_model.predict(X[:20])
        np.testing.assert_array_equal(pred1, pred2,
                                       err_msg="Model should be deterministic")
