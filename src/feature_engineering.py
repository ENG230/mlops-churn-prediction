"""
Feature Engineering Module
===========================
Handles data preprocessing, feature transformations, train/test splitting,
and saving processed datasets.

MLOps Concept: Feature engineering is versioned and reproducible.
The same transformations applied during training MUST be applied at inference.
We use sklearn Pipelines to ensure this consistency.
"""

import logging
import pandas as pd
import numpy as np
import yaml
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def load_config(config_path: str = "configs/params.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def preprocess_raw_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Clean and preprocess raw data before feature engineering.
    
    Steps:
    - Drop unnecessary columns
    - Handle TotalCharges (can be empty string for new customers)
    - Encode target variable
    """
    logger.info("Preprocessing raw data...")
    df = df.copy()

    # Drop columns not used for modeling
    drop_cols = config["features"]["drop_columns"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    logger.info(f"Dropped columns: {drop_cols}")

    # Fix TotalCharges - convert to numeric (new customers may have empty string)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        null_count = df["TotalCharges"].isnull().sum()
        if null_count > 0:
            logger.info(f"Filling {null_count} null TotalCharges with 0 (new customers)")
            df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Encode target: Yes -> 1, No -> 0
    target = config["data"]["target_column"]
    if df[target].dtype == object:
        df[target] = (df[target] == "Yes").astype(int)
        logger.info(f"Target encoded: Yes=1, No=0")

    logger.info(f"Preprocessing complete. Shape: {df.shape}")
    return df


def build_preprocessor(config: dict) -> ColumnTransformer:
    """
    Build sklearn ColumnTransformer for feature preprocessing.
    
    MLOps Concept: Using sklearn Pipeline ensures the EXACT same
    transformations are applied during training and inference.
    The fitted preprocessor is saved as an artifact.
    
    Pipeline steps:
    - Numerical: Impute missing → StandardScaler
    - Categorical: Impute missing → OneHotEncoder
    """
    numerical_features = config["features"]["numerical_features"]
    categorical_features = config["features"]["categorical_features"]

    # Numerical pipeline
    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Categorical pipeline
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    # Combine into ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, numerical_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",  # Drop any columns not specified
        verbose_feature_names_out=True,
    )

    logger.info("Preprocessor pipeline built:")
    logger.info(f"  Numerical features ({len(numerical_features)}): {numerical_features}")
    logger.info(f"  Categorical features ({len(categorical_features)}): {categorical_features}")
    return preprocessor


def engineer_features(
    df: pd.DataFrame,
    config: dict,
    preprocessor=None,
    fit: bool = True,
    save_preprocessor: bool = True,
    preprocessor_path: str = "models/preprocessor.joblib"
) -> tuple:
    """
    Apply feature engineering and return processed arrays.
    
    Args:
        df: Input DataFrame (already preprocessed)
        config: Configuration dictionary
        preprocessor: Existing preprocessor (for inference)
        fit: Whether to fit the preprocessor (True for training, False for inference)
        save_preprocessor: Whether to save the fitted preprocessor
        preprocessor_path: Path to save/load preprocessor
    
    Returns:
        X: Feature matrix (numpy array)
        y: Target vector (numpy array)
        preprocessor: Fitted ColumnTransformer
        feature_names: List of feature names after transformation
    """
    target = config["data"]["target_column"]
    numerical_features = config["features"]["numerical_features"]
    categorical_features = config["features"]["categorical_features"]

    # Separate features and target
    feature_cols = numerical_features + categorical_features
    X_raw = df[feature_cols]
    y = df[target].values

    if fit:
        # Build and fit preprocessor
        preprocessor = build_preprocessor(config)
        X = preprocessor.fit_transform(X_raw)
        logger.info(f"Preprocessor fitted on {len(X_raw)} samples")

        if save_preprocessor:
            Path(preprocessor_path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(preprocessor, preprocessor_path)
            logger.info(f"Preprocessor saved to {preprocessor_path}")
    else:
        # Use existing preprocessor for inference
        if preprocessor is None:
            logger.info(f"Loading preprocessor from {preprocessor_path}")
            preprocessor = joblib.load(preprocessor_path)
        X = preprocessor.transform(X_raw)
        logger.info(f"Preprocessor applied to {len(X_raw)} samples")

    # Get feature names after transformation
    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Target distribution: {pd.Series(y).value_counts().to_dict()}")
    return X, y, preprocessor, feature_names


def split_and_save_data(
    df: pd.DataFrame,
    config: dict,
    save: bool = True
) -> tuple:
    """
    Split data into train/test sets and optionally save to disk.
    
    MLOps Concept: We save a 'reference' dataset (training data)
    for later use in drift detection monitoring.
    """
    target = config["data"]["target_column"]
    test_size = config["data"]["test_size"]
    random_state = config["data"]["random_state"]

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target]  # Stratify to maintain class balance
    )

    logger.info(f"Train/Test split: {len(train_df)} train, {len(test_df)} test")
    logger.info(f"Train churn rate: {train_df[target].mean():.2%}")
    logger.info(f"Test churn rate: {test_df[target].mean():.2%}")

    if save:
        train_path = config["data"]["processed_train_path"]
        test_path = config["data"]["processed_test_path"]
        ref_path = config["data"]["processed_reference_path"]

        Path(train_path).parent.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        # Reference dataset = training data (used for drift detection baseline)
        train_df.to_csv(ref_path, index=False)

        logger.info(f"Train data saved to {train_path}")
        logger.info(f"Test data saved to {test_path}")
        logger.info(f"Reference data saved to {ref_path}")

    return train_df, test_df


def run_feature_engineering(config_path: str = "configs/params.yaml"):
    """
    Full feature engineering pipeline:
    1. Load raw data
    2. Preprocess
    3. Split into train/test
    4. Fit and save preprocessor
    5. Return processed arrays
    """
    from src.data_ingestion import ingest_data

    config = load_config(config_path)

    # Step 1: Ingest data
    raw_df = ingest_data(config_path)

    # Step 2: Preprocess
    processed_df = preprocess_raw_data(raw_df, config)

    # Step 3: Split
    train_df, test_df = split_and_save_data(processed_df, config)

    # Step 4 & 5: Engineer features (fit on train, transform both)
    X_train, y_train, preprocessor, feature_names = engineer_features(
        train_df, config, fit=True
    )
    X_test, y_test, _, _ = engineer_features(
        test_df, config, preprocessor=preprocessor, fit=False
    )

    logger.info("✅ Feature engineering complete!")
    logger.info(f"   X_train: {X_train.shape}, y_train: {y_train.shape}")
    logger.info(f"   X_test: {X_test.shape}, y_test: {y_test.shape}")
    logger.info(f"   Total features: {len(feature_names)}")

    return X_train, y_train, X_test, y_test, preprocessor, feature_names


if __name__ == "__main__":
    X_train, y_train, X_test, y_test, preprocessor, feature_names = run_feature_engineering()
    print(f"\nFeature Engineering Complete!")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Number of features: {len(feature_names)}")
    print(f"\nSample feature names: {feature_names[:10]}")
