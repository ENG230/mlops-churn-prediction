"""
Data Ingestion Module
=====================
Handles loading raw data, generating synthetic data for demo,
and performing initial data validation checks.

MLOps Concept: Data is the foundation. We validate schema,
check for nulls, and ensure data quality before any processing.
"""

import os
import logging
import pandas as pd
import numpy as np
import yaml
from pathlib import Path

# Configure logging - MLOps best practice: structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/params.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Configuration loaded from {config_path}")
    return config


def generate_synthetic_churn_data(n_samples: int = 7043, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic Telco Customer Churn dataset.
    
    This mimics the real Kaggle Telco Churn dataset structure.
    In a real project, this would be replaced by actual data loading.
    
    MLOps Note: We generate data programmatically so the project
    runs without external dependencies (no Kaggle API needed).
    """
    np.random.seed(random_state)
    n = n_samples

    logger.info(f"Generating synthetic churn dataset with {n} samples...")

    # Customer IDs
    customer_ids = [f"CUST-{i:05d}" for i in range(1, n + 1)]

    # Demographics
    gender = np.random.choice(["Male", "Female"], n)
    senior_citizen = np.random.choice([0, 1], n, p=[0.84, 0.16])
    partner = np.random.choice(["Yes", "No"], n, p=[0.48, 0.52])
    dependents = np.random.choice(["Yes", "No"], n, p=[0.30, 0.70])

    # Service features
    tenure = np.random.randint(0, 73, n)
    phone_service = np.random.choice(["Yes", "No"], n, p=[0.90, 0.10])
    multiple_lines = np.where(
        phone_service == "No", "No phone service",
        np.random.choice(["Yes", "No"], n, p=[0.42, 0.58])
    )
    internet_service = np.random.choice(
        ["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22]
    )
    online_security = np.where(
        internet_service == "No", "No internet service",
        np.random.choice(["Yes", "No"], n, p=[0.29, 0.71])
    )
    online_backup = np.where(
        internet_service == "No", "No internet service",
        np.random.choice(["Yes", "No"], n, p=[0.34, 0.66])
    )
    device_protection = np.where(
        internet_service == "No", "No internet service",
        np.random.choice(["Yes", "No"], n, p=[0.34, 0.66])
    )
    tech_support = np.where(
        internet_service == "No", "No internet service",
        np.random.choice(["Yes", "No"], n, p=[0.29, 0.71])
    )
    streaming_tv = np.where(
        internet_service == "No", "No internet service",
        np.random.choice(["Yes", "No"], n, p=[0.38, 0.62])
    )
    streaming_movies = np.where(
        internet_service == "No", "No internet service",
        np.random.choice(["Yes", "No"], n, p=[0.39, 0.61])
    )

    # Contract & billing
    contract = np.random.choice(
        ["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24]
    )
    paperless_billing = np.random.choice(["Yes", "No"], n, p=[0.59, 0.41])
    payment_method = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        n, p=[0.34, 0.23, 0.22, 0.21]
    )

    # Charges - correlated with tenure and services
    monthly_charges = np.round(
        20 + (internet_service == "Fiber optic") * 30
        + (internet_service == "DSL") * 15
        + (phone_service == "Yes") * 10
        + np.random.normal(0, 5, n), 2
    )
    monthly_charges = np.clip(monthly_charges, 18.25, 118.75)
    total_charges = np.round(monthly_charges * tenure + np.random.normal(0, 10, n), 2)
    total_charges = np.clip(total_charges, 0, None)

    # Target: Churn - influenced by contract type, tenure, charges
    churn_prob = (
        0.05
        + (contract == "Month-to-month") * 0.25
        + (tenure < 12) * 0.15
        + (internet_service == "Fiber optic") * 0.10
        + (monthly_charges > 70) * 0.10
        - (tenure > 36) * 0.10
        - (contract == "Two year") * 0.15
    )
    churn_prob = np.clip(churn_prob, 0.02, 0.95)
    churn = np.where(np.random.random(n) < churn_prob, "Yes", "No")

    df = pd.DataFrame({
        "customerID": customer_ids,
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Churn": churn,
    })

    logger.info(f"Dataset generated: {df.shape[0]} rows, {df.shape[1]} columns")
    logger.info(f"Churn rate: {(df['Churn'] == 'Yes').mean():.2%}")
    return df


def validate_data(df: pd.DataFrame, config: dict) -> bool:
    """
    Validate data schema and quality.
    
    MLOps Concept: Data validation is a critical gate before training.
    Catching bad data early prevents silent model failures.
    """
    logger.info("Running data validation checks...")
    errors = []

    # Check 1: Required columns exist
    required_cols = (
        config["features"]["numerical_features"]
        + config["features"]["categorical_features"]
        + [config["data"]["target_column"]]
    )
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    # Check 2: No completely empty columns
    empty_cols = df.columns[df.isnull().all()].tolist()
    if empty_cols:
        errors.append(f"Completely empty columns: {empty_cols}")

    # Check 3: Target column has expected values
    target = config["data"]["target_column"]
    if target in df.columns:
        unexpected_target = set(df[target].unique()) - {"Yes", "No"}
        if unexpected_target:
            errors.append(f"Unexpected target values: {unexpected_target}")

    # Check 4: Minimum row count
    if len(df) < 100:
        errors.append(f"Dataset too small: {len(df)} rows (minimum 100)")

    # Check 5: Numerical features are numeric
    for col in config["features"]["numerical_features"]:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"Column '{col}' should be numeric but is {df[col].dtype}")

    # Report results
    if errors:
        for err in errors:
            logger.error(f"Validation FAILED: {err}")
        return False

    # Log data statistics
    null_pct = df.isnull().mean().mean() * 100
    logger.info(f"✅ Data validation PASSED")
    logger.info(f"   Rows: {len(df):,} | Columns: {len(df.columns)}")
    logger.info(f"   Null percentage: {null_pct:.2f}%")
    logger.info(f"   Target distribution: {df[target].value_counts().to_dict()}")
    return True


def ingest_data(config_path: str = "configs/params.yaml") -> pd.DataFrame:
    """
    Main data ingestion function.
    
    Loads or generates data, validates it, and saves to raw data path.
    """
    config = load_config(config_path)
    raw_path = config["data"]["raw_data_path"]

    # Create directory if it doesn't exist
    Path(raw_path).parent.mkdir(parents=True, exist_ok=True)

    # Load existing data or generate synthetic data
    if os.path.exists(raw_path):
        logger.info(f"Loading existing data from {raw_path}")
        df = pd.read_csv(raw_path)
    else:
        logger.info("No existing data found. Generating synthetic dataset...")
        df = generate_synthetic_churn_data(
            random_state=config["data"]["random_state"]
        )
        df.to_csv(raw_path, index=False)
        logger.info(f"Raw data saved to {raw_path}")

    # Validate data
    is_valid = validate_data(df, config)
    if not is_valid:
        raise ValueError("Data validation failed. Check logs for details.")

    return df


if __name__ == "__main__":
    df = ingest_data()
    print(f"\nData ingestion complete!")
    print(f"Shape: {df.shape}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nChurn distribution:")
    print(df["Churn"].value_counts(normalize=True))
