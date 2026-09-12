"""
Data Validation Tests
======================
Tests for data ingestion, validation, and feature engineering.

MLOps Concept: Testing is a first-class citizen in MLOps.
Data tests catch issues before they silently corrupt model training.

Test Categories:
- Schema tests: correct columns, types
- Statistical tests: distributions, ranges
- Pipeline tests: preprocessing correctness
"""

import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_ingestion import generate_synthetic_churn_data, validate_data, load_config
from src.feature_engineering import preprocess_raw_data, build_preprocessor


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def config():
    """Load test configuration."""
    return load_config("configs/params.yaml")


@pytest.fixture
def sample_data():
    """Generate a small sample dataset for testing."""
    return generate_synthetic_churn_data(n_samples=500, random_state=42)


@pytest.fixture
def processed_data(sample_data, config):
    """Return preprocessed data."""
    return preprocess_raw_data(sample_data, config)


# ─────────────────────────────────────────────────────────────
# Data Schema Tests
# ─────────────────────────────────────────────────────────────

class TestDataSchema:
    """Tests for data schema correctness."""

    def test_dataset_has_correct_shape(self, sample_data):
        """Dataset should have expected number of columns."""
        assert sample_data.shape[0] == 500, "Should have 500 rows"
        assert sample_data.shape[1] == 21, f"Should have 21 columns, got {sample_data.shape[1]}"

    def test_required_columns_exist(self, sample_data, config):
        """All required feature columns must be present."""
        required = (
            config["features"]["numerical_features"]
            + config["features"]["categorical_features"]
            + [config["data"]["target_column"]]
        )
        for col in required:
            assert col in sample_data.columns, f"Missing required column: {col}"

    def test_target_column_values(self, sample_data, config):
        """Target column should only contain 'Yes' or 'No'."""
        target = config["data"]["target_column"]
        valid_values = {"Yes", "No"}
        actual_values = set(sample_data[target].unique())
        assert actual_values.issubset(valid_values), \
            f"Unexpected target values: {actual_values - valid_values}"

    def test_numerical_features_are_numeric(self, sample_data, config):
        """Numerical features should have numeric dtype."""
        for col in config["features"]["numerical_features"]:
            assert pd.api.types.is_numeric_dtype(sample_data[col]), \
                f"Column '{col}' should be numeric, got {sample_data[col].dtype}"

    def test_no_completely_empty_columns(self, sample_data):
        """No column should be entirely null."""
        empty_cols = sample_data.columns[sample_data.isnull().all()].tolist()
        assert len(empty_cols) == 0, f"Completely empty columns: {empty_cols}"

    def test_customer_id_is_unique(self, sample_data):
        """CustomerID should be unique."""
        assert sample_data["customerID"].nunique() == len(sample_data), \
            "CustomerID values are not unique"


# ─────────────────────────────────────────────────────────────
# Data Quality Tests
# ─────────────────────────────────────────────────────────────

class TestDataQuality:
    """Tests for data quality and statistical properties."""

    def test_tenure_range(self, sample_data):
        """Tenure should be between 0 and 72 months."""
        assert sample_data["tenure"].min() >= 0, "Tenure cannot be negative"
        assert sample_data["tenure"].max() <= 72, "Tenure exceeds maximum (72 months)"

    def test_monthly_charges_range(self, sample_data):
        """Monthly charges should be within reasonable bounds."""
        assert sample_data["MonthlyCharges"].min() >= 0, "Monthly charges cannot be negative"
        assert sample_data["MonthlyCharges"].max() <= 200, "Monthly charges seem too high"

    def test_total_charges_non_negative(self, sample_data):
        """Total charges should be non-negative."""
        assert (sample_data["TotalCharges"] >= 0).all(), "TotalCharges has negative values"

    def test_churn_rate_reasonable(self, sample_data, config):
        """Churn rate should be between 5% and 50% (business sanity check)."""
        target = config["data"]["target_column"]
        churn_rate = (sample_data[target] == "Yes").mean()
        assert 0.05 <= churn_rate <= 0.50, \
            f"Churn rate {churn_rate:.2%} is outside expected range [5%, 50%]"

    def test_senior_citizen_binary(self, sample_data):
        """SeniorCitizen should only be 0 or 1."""
        valid_values = {0, 1}
        actual_values = set(sample_data["SeniorCitizen"].unique())
        assert actual_values.issubset(valid_values), \
            f"SeniorCitizen has unexpected values: {actual_values}"

    def test_gender_values(self, sample_data):
        """Gender should only be Male or Female."""
        valid_values = {"Male", "Female"}
        actual_values = set(sample_data["gender"].unique())
        assert actual_values.issubset(valid_values), \
            f"Gender has unexpected values: {actual_values}"

    def test_contract_values(self, sample_data):
        """Contract should have valid values."""
        valid_values = {"Month-to-month", "One year", "Two year"}
        actual_values = set(sample_data["Contract"].unique())
        assert actual_values.issubset(valid_values), \
            f"Contract has unexpected values: {actual_values - valid_values}"

    def test_internet_service_values(self, sample_data):
        """InternetService should have valid values."""
        valid_values = {"DSL", "Fiber optic", "No"}
        actual_values = set(sample_data["InternetService"].unique())
        assert actual_values.issubset(valid_values), \
            f"InternetService has unexpected values: {actual_values - valid_values}"

    def test_null_percentage_acceptable(self, sample_data):
        """Overall null percentage should be below 5%."""
        null_pct = sample_data.isnull().mean().mean()
        assert null_pct < 0.05, f"Too many nulls: {null_pct:.2%}"


# ─────────────────────────────────────────────────────────────
# Data Validation Function Tests
# ─────────────────────────────────────────────────────────────

class TestDataValidation:
    """Tests for the validate_data function."""

    def test_valid_data_passes_validation(self, sample_data, config):
        """Valid data should pass all validation checks."""
        result = validate_data(sample_data, config)
        assert result is True, "Valid data should pass validation"

    def test_missing_column_fails_validation(self, sample_data, config):
        """Data with missing required column should fail validation."""
        bad_data = sample_data.drop(columns=["tenure"])
        result = validate_data(bad_data, config)
        assert result is False, "Data with missing column should fail validation"

    def test_too_few_rows_fails_validation(self, config):
        """Dataset with fewer than 100 rows should fail validation."""
        tiny_data = generate_synthetic_churn_data(n_samples=50, random_state=42)
        result = validate_data(tiny_data, config)
        assert result is False, "Tiny dataset should fail validation"

    def test_invalid_target_values_fails_validation(self, sample_data, config):
        """Data with invalid target values should fail validation."""
        bad_data = sample_data.copy()
        bad_data["Churn"] = "Maybe"  # Invalid value
        result = validate_data(bad_data, config)
        assert result is False, "Invalid target values should fail validation"


# ─────────────────────────────────────────────────────────────
# Preprocessing Tests
# ─────────────────────────────────────────────────────────────

class TestPreprocessing:
    """Tests for data preprocessing steps."""

    def test_customer_id_dropped(self, processed_data):
        """customerID should be dropped after preprocessing."""
        assert "customerID" not in processed_data.columns, \
            "customerID should be dropped during preprocessing"

    def test_target_encoded_as_binary(self, processed_data, config):
        """Target should be encoded as 0/1 after preprocessing."""
        target = config["data"]["target_column"]
        valid_values = {0, 1}
        actual_values = set(processed_data[target].unique())
        assert actual_values.issubset(valid_values), \
            f"Target should be 0/1, got: {actual_values}"

    def test_total_charges_numeric(self, processed_data):
        """TotalCharges should be numeric after preprocessing."""
        assert pd.api.types.is_numeric_dtype(processed_data["TotalCharges"]), \
            "TotalCharges should be numeric after preprocessing"

    def test_no_nulls_in_numerical_after_preprocessing(self, processed_data, config):
        """Numerical features should have no nulls after preprocessing."""
        for col in config["features"]["numerical_features"]:
            if col in processed_data.columns:
                null_count = processed_data[col].isnull().sum()
                assert null_count == 0, \
                    f"Column '{col}' has {null_count} nulls after preprocessing"

    def test_preprocessor_builds_successfully(self, config):
        """Preprocessor pipeline should build without errors."""
        preprocessor = build_preprocessor(config)
        assert preprocessor is not None, "Preprocessor should not be None"

    def test_preprocessor_transforms_data(self, processed_data, config):
        """Preprocessor should transform data to expected shape."""
        from src.feature_engineering import engineer_features
        X, y, preprocessor, feature_names = engineer_features(
            processed_data, config, fit=True, save_preprocessor=False
        )
        assert X is not None, "Transformed X should not be None"
        assert len(X) == len(processed_data), "Row count should be preserved"
        assert X.shape[1] > 0, "Should have at least one feature"
        assert len(feature_names) == X.shape[1], \
            "Feature names count should match feature matrix columns"

    def test_feature_matrix_no_nan(self, processed_data, config):
        """Feature matrix should have no NaN values after transformation."""
        from src.feature_engineering import engineer_features
        X, y, _, _ = engineer_features(
            processed_data, config, fit=True, save_preprocessor=False
        )
        assert not np.isnan(X).any(), "Feature matrix should not contain NaN values"

    def test_feature_matrix_no_inf(self, processed_data, config):
        """Feature matrix should have no infinite values."""
        from src.feature_engineering import engineer_features
        X, y, _, _ = engineer_features(
            processed_data, config, fit=True, save_preprocessor=False
        )
        assert not np.isinf(X).any(), "Feature matrix should not contain infinite values"


# ─────────────────────────────────────────────────────────────
# Data Reproducibility Tests
# ─────────────────────────────────────────────────────────────

class TestReproducibility:
    """Tests for data reproducibility (same seed = same data)."""

    def test_same_seed_produces_same_data(self):
        """Same random seed should produce identical datasets."""
        df1 = generate_synthetic_churn_data(n_samples=100, random_state=42)
        df2 = generate_synthetic_churn_data(n_samples=100, random_state=42)
        pd.testing.assert_frame_equal(df1, df2), \
            "Same seed should produce identical data"

    def test_different_seeds_produce_different_data(self):
        """Different seeds should produce different datasets."""
        df1 = generate_synthetic_churn_data(n_samples=100, random_state=42)
        df2 = generate_synthetic_churn_data(n_samples=100, random_state=99)
        assert not df1.equals(df2), "Different seeds should produce different data"
