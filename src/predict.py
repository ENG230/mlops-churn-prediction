"""
Prediction / Inference Module
==============================
Handles loading the trained model and making predictions.

MLOps Concepts Demonstrated:
- Loading model from MLflow Model Registry
- Consistent preprocessing at inference time
- Prediction with confidence scores
- Input validation before inference
"""

import logging
import yaml
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def load_config(config_path: str = "configs/params.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_model_from_registry(
    model_name: str,
    stage: str = "Production",
    tracking_uri: str = "mlruns"
):
    """
    Load model from MLflow Model Registry.
    
    MLOps Concept: Models are loaded by NAME and STAGE, not by file path.
    This decouples the serving layer from the training layer.
    Stages: None → Staging → Production → Archived
    """
    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"models:/{model_name}/{stage}"

    try:
        model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"Model loaded from registry: {model_uri}")
        return model
    except Exception as e:
        logger.warning(f"Could not load from registry ({e}). Trying 'latest' version...")
        try:
            model_uri = f"models:/{model_name}/latest"
            model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"Model loaded: {model_uri}")
            return model
        except Exception as e2:
            logger.error(f"Failed to load from registry: {e2}")
            return None


def load_model_from_file(model_path: str = "models/model.joblib"):
    """Fallback: load model directly from file."""
    if Path(model_path).exists():
        model = joblib.load(model_path)
        logger.info(f"Model loaded from file: {model_path}")
        return model
    logger.error(f"Model file not found: {model_path}")
    return None


def load_preprocessor(preprocessor_path: str = "models/preprocessor.joblib"):
    """Load the fitted preprocessor."""
    if Path(preprocessor_path).exists():
        preprocessor = joblib.load(preprocessor_path)
        logger.info(f"Preprocessor loaded from: {preprocessor_path}")
        return preprocessor
    logger.error(f"Preprocessor not found: {preprocessor_path}")
    return None


def validate_input(data: pd.DataFrame, config: dict) -> tuple:
    """
    Validate input data before inference.
    
    Returns (is_valid, error_message)
    """
    numerical_features = config["features"]["numerical_features"]
    categorical_features = config["features"]["categorical_features"]
    required_features = numerical_features + categorical_features

    missing = [f for f in required_features if f not in data.columns]
    if missing:
        return False, f"Missing required features: {missing}"

    # Check for all-null rows
    feature_data = data[required_features]
    all_null_rows = feature_data.isnull().all(axis=1).sum()
    if all_null_rows > 0:
        return False, f"{all_null_rows} rows have all null values"

    return True, None


def predict(
    data: Union[pd.DataFrame, dict],
    config_path: str = "configs/params.yaml",
    model=None,
    preprocessor=None,
    return_proba: bool = True,
) -> pd.DataFrame:
    """
    Make predictions on input data.
    
    Args:
        data: Input DataFrame or dict with feature values
        config_path: Path to config file
        model: Pre-loaded model (optional, loads from registry if None)
        preprocessor: Pre-loaded preprocessor (optional)
        return_proba: Whether to return probability scores
    
    Returns:
        DataFrame with predictions and confidence scores
    """
    config = load_config(config_path)

    # Convert dict to DataFrame
    if isinstance(data, dict):
        data = pd.DataFrame([data])

    # Validate input
    is_valid, error_msg = validate_input(data, config)
    if not is_valid:
        raise ValueError(f"Input validation failed: {error_msg}")

    # Load model if not provided
    if model is None:
        model = load_model_from_registry(
            model_name=config["mlflow"]["model_name"],
            tracking_uri=config["mlflow"]["tracking_uri"]
        )
        if model is None:
            raise RuntimeError("Could not load model. Run training first.")

    # Load preprocessor if not provided
    if preprocessor is None:
        preprocessor = load_preprocessor()
        if preprocessor is None:
            raise RuntimeError("Could not load preprocessor. Run training first.")

    # Preprocess input
    from src.feature_engineering import preprocess_raw_data, engineer_features

    # Handle target column - add dummy if not present (inference scenario)
    target = config["data"]["target_column"]
    data_copy = data.copy()
    if target not in data_copy.columns:
        data_copy[target] = 0  # Dummy target for preprocessing

    # Drop customerID if present
    if "customerID" in data_copy.columns:
        data_copy = data_copy.drop(columns=["customerID"])

    # Preprocess
    data_copy = preprocess_raw_data(data_copy, config)

    # Transform features
    X, _, _, _ = engineer_features(
        data_copy, config,
        preprocessor=preprocessor,
        fit=False,
        save_preprocessor=False
    )

    # Make predictions
    predictions = model.predict(X)
    results = pd.DataFrame({
        "prediction": predictions,
        "prediction_label": ["Churn" if p == 1 else "No Churn" for p in predictions],
    })

    if return_proba and hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        results["churn_probability"] = probabilities[:, 1].round(4)
        results["no_churn_probability"] = probabilities[:, 0].round(4)
        results["confidence"] = np.max(probabilities, axis=1).round(4)

    logger.info(f"Predictions made for {len(data)} samples")
    logger.info(f"Churn predictions: {predictions.sum()} / {len(predictions)}")

    return results


def batch_predict(
    input_path: str,
    output_path: str,
    config_path: str = "configs/params.yaml"
) -> pd.DataFrame:
    """
    Batch prediction on a CSV file.
    
    MLOps Concept: Batch inference for offline scoring of large datasets.
    Results are saved with predictions appended to original data.
    """
    logger.info(f"Running batch prediction on {input_path}")

    # Load data
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} records for batch prediction")

    # Make predictions
    results = predict(df, config_path=config_path)

    # Combine original data with predictions
    output_df = pd.concat([df.reset_index(drop=True), results], axis=1)

    # Save results
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    logger.info(f"Batch predictions saved to {output_path}")
    logger.info(f"Churn rate in predictions: {results['prediction'].mean():.2%}")

    return output_df


if __name__ == "__main__":
    # Example: Single prediction
    sample_customer = {
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

    print("Making prediction for sample customer...")
    result = predict(sample_customer)
    print(f"\nPrediction Result:")
    print(result.to_string(index=False))
