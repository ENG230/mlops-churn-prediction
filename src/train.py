"""
Model Training Module
=====================
Trains ML models with full MLflow experiment tracking.
Logs parameters, metrics, artifacts, and registers the model.

MLOps Concepts Demonstrated:
- Experiment tracking (every run is logged)
- Hyperparameter management (from config file)
- Model versioning (MLflow Model Registry)
- Artifact logging (model, preprocessor, plots)
- Reproducibility (random seeds, versioned configs)
"""

import logging
import yaml
import json
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def load_config(config_path: str = "configs/params.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_model(config: dict):
    """
    Factory function to instantiate the model based on config.
    
    MLOps Concept: Algorithm selection is config-driven,
    not hardcoded. Change params.yaml to switch algorithms.
    """
    algorithm = config["model"]["algorithm"]
    logger.info(f"Instantiating model: {algorithm}")

    if algorithm == "random_forest":
        params = config["model"]["random_forest"]
        model = RandomForestClassifier(**params)

    elif algorithm == "xgboost":
        if not XGBOOST_AVAILABLE:
            logger.warning("XGBoost not available, falling back to RandomForest")
            params = config["model"]["random_forest"]
            model = RandomForestClassifier(**params)
            algorithm = "random_forest"
        else:
            params = config["model"]["xgboost"]
            # Remove non-XGBoost params
            params = {k: v for k, v in params.items()
                      if k not in ["use_label_encoder"]}
            model = XGBClassifier(**params)

    elif algorithm == "logistic_regression":
        params = config["model"]["logistic_regression"]
        model = LogisticRegression(**params)

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. "
                         f"Choose from: random_forest, xgboost, logistic_regression")

    return model, algorithm, params


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute all evaluation metrics."""
    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
    }
    return metrics


def plot_confusion_matrix(y_true, y_pred, output_path: str = "reports/confusion_matrix.png"):
    """Generate and save confusion matrix plot."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
        ax=ax
    )
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_xlabel("Predicted", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix saved to {output_path}")
    return output_path


def plot_roc_curve(y_true, y_prob, output_path: str = "reports/roc_curve.png"):
    """Generate and save ROC curve plot."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.1, color="darkorange")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve - Churn Prediction", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()
    logger.info(f"ROC curve saved to {output_path}")
    return output_path


def plot_feature_importance(
    model, feature_names: list, top_n: int = 20,
    output_path: str = "reports/feature_importance.png"
):
    """Generate and save feature importance plot."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Get feature importances
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        logger.warning("Model does not support feature importance")
        return None

    # Sort and select top N
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] if i < len(feature_names) else f"f{i}"
                    for i in indices]
    top_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_features)))
    bars = ax.barh(range(len(top_features)), top_importances[::-1], color=colors[::-1])
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features[::-1], fontsize=9)
    ax.set_xlabel("Feature Importance", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()
    logger.info(f"Feature importance plot saved to {output_path}")
    return output_path


def check_model_quality_gate(metrics: dict, config: dict) -> bool:
    """
    MLOps Concept: Quality Gate
    
    Before registering a model, it must pass minimum performance thresholds.
    This prevents deploying degraded models automatically.
    In CI/CD, a failed quality gate stops the pipeline.
    """
    thresholds = config["evaluation"]
    passed = True
    results = []

    checks = [
        ("accuracy", thresholds["min_accuracy_threshold"]),
        ("f1", thresholds["min_f1_threshold"]),
        ("roc_auc", thresholds["min_roc_auc_threshold"]),
    ]

    for metric_name, threshold in checks:
        value = metrics.get(metric_name, 0)
        status = "✅ PASS" if value >= threshold else "❌ FAIL"
        if value < threshold:
            passed = False
        results.append(f"  {status} | {metric_name}: {value:.4f} (threshold: {threshold})")

    logger.info("=" * 50)
    logger.info("MODEL QUALITY GATE RESULTS:")
    for r in results:
        logger.info(r)
    logger.info(f"Overall: {'✅ PASSED' if passed else '❌ FAILED'}")
    logger.info("=" * 50)

    return passed


def train_model(config_path: str = "configs/params.yaml") -> dict:
    """
    Main training function with full MLflow tracking.
    
    MLOps Flow:
    1. Load config (versioned hyperparameters)
    2. Load processed data
    3. Start MLflow run
    4. Log all parameters
    5. Train model
    6. Evaluate and log metrics
    7. Log artifacts (model, plots, preprocessor)
    8. Check quality gate
    9. Register model in MLflow Model Registry
    """
    config = load_config(config_path)

    # Setup MLflow
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    logger.info("=" * 60)
    logger.info("STARTING MODEL TRAINING")
    logger.info("=" * 60)

    # Load processed data
    train_path = config["data"]["processed_train_path"]
    test_path = config["data"]["processed_test_path"]

    if not Path(train_path).exists():
        logger.info("Processed data not found. Running feature engineering...")
        from src.feature_engineering import run_feature_engineering
        X_train, y_train, X_test, y_test, preprocessor, feature_names = \
            run_feature_engineering(config_path)
    else:
        from src.feature_engineering import (
            preprocess_raw_data, engineer_features, load_config as fe_load_config
        )
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        train_df = preprocess_raw_data(train_df, config)
        test_df = preprocess_raw_data(test_df, config)

        X_train, y_train, preprocessor, feature_names = engineer_features(
            train_df, config, fit=True
        )
        X_test, y_test, _, _ = engineer_features(
            test_df, config, preprocessor=preprocessor, fit=False
        )

    # Get model
    model, algorithm, model_params = get_model(config)

    # Start MLflow run
    run_name = f"{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        logger.info(f"MLflow Run ID: {run_id}")
        logger.info(f"MLflow Run Name: {run_name}")

        # ── Log Parameters ──────────────────────────────────────
        mlflow.log_param("algorithm", algorithm)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("test_size", config["data"]["test_size"])
        mlflow.log_param("random_state", config["data"]["random_state"])

        for param_name, param_value in model_params.items():
            mlflow.log_param(f"model_{param_name}", param_value)

        # ── Train Model ──────────────────────────────────────────
        logger.info(f"Training {algorithm} model...")
        model.fit(X_train, y_train)
        logger.info("Training complete!")

        # ── Evaluate Model ───────────────────────────────────────
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        y_prob_test = model.predict_proba(X_test)[:, 1]

        train_metrics = compute_metrics(y_train, y_pred_train,
                                        model.predict_proba(X_train)[:, 1])
        test_metrics = compute_metrics(y_test, y_pred_test, y_prob_test)

        # ── Log Metrics ──────────────────────────────────────────
        for metric_name, value in train_metrics.items():
            mlflow.log_metric(f"train_{metric_name}", value)
        for metric_name, value in test_metrics.items():
            mlflow.log_metric(f"test_{metric_name}", value)

        logger.info("TRAINING METRICS:")
        for k, v in train_metrics.items():
            logger.info(f"  train_{k}: {v:.4f}")
        logger.info("TEST METRICS:")
        for k, v in test_metrics.items():
            logger.info(f"  test_{k}: {v:.4f}")

        # ── Generate & Log Plots ─────────────────────────────────
        cm_path = plot_confusion_matrix(y_test, y_pred_test)
        roc_path = plot_roc_curve(y_test, y_prob_test)
        fi_path = plot_feature_importance(model, feature_names)

        mlflow.log_artifact(cm_path, "plots")
        mlflow.log_artifact(roc_path, "plots")
        if fi_path:
            mlflow.log_artifact(fi_path, "plots")

        # ── Log Classification Report ────────────────────────────
        report = classification_report(y_test, y_pred_test,
                                       target_names=["No Churn", "Churn"])
        report_path = "reports/classification_report.txt"
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            f.write(f"Algorithm: {algorithm}\n")
            f.write(f"Run ID: {run_id}\n\n")
            f.write(report)
        mlflow.log_artifact(report_path)
        logger.info(f"\nClassification Report:\n{report}")

        # ── Log Config as Artifact ───────────────────────────────
        mlflow.log_artifact(config_path, "config")

        # ── Log Preprocessor ─────────────────────────────────────
        preprocessor_path = "models/preprocessor.joblib"
        if Path(preprocessor_path).exists():
            mlflow.log_artifact(preprocessor_path, "preprocessor")

        # ── Log Model ────────────────────────────────────────────
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=X_train[:5],
        )
        # Register model separately (compatible with MLflow 2.x and 3.x)
        try:
            model_uri = f"runs:/{run_id}/model"
            mlflow.register_model(model_uri, config["mlflow"]["model_name"])
        except Exception as reg_err:
            logger.warning(f"Model registration skipped: {reg_err}")
        logger.info(f"Model logged to MLflow")

        # ── Quality Gate ─────────────────────────────────────────
        quality_passed = check_model_quality_gate(test_metrics, config)
        mlflow.log_param("quality_gate_passed", quality_passed)

        if quality_passed:
            mlflow.set_tag("quality_gate", "PASSED")
            mlflow.set_tag("model_status", "candidate")
        else:
            mlflow.set_tag("quality_gate", "FAILED")
            mlflow.set_tag("model_status", "rejected")

        # ── Save metrics to file (for DVC tracking) ──────────────
        metrics_path = "reports/metrics.json"
        all_metrics = {
            "train": train_metrics,
            "test": test_metrics,
            "quality_gate_passed": quality_passed,
            "run_id": run_id,
            "algorithm": algorithm,
        }
        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")

        logger.info("=" * 60)
        logger.info(f"✅ Training complete! Run ID: {run_id}")
        logger.info(f"   View in MLflow UI: mlflow ui --port 5000")
        logger.info("=" * 60)

        return {
            "run_id": run_id,
            "model": model,
            "metrics": test_metrics,
            "quality_gate_passed": quality_passed,
            "feature_names": feature_names,
        }


if __name__ == "__main__":
    result = train_model()
    print(f"\n{'='*50}")
    print(f"Training Complete!")
    print(f"Run ID: {result['run_id']}")
    print(f"Test Metrics: {result['metrics']}")
    print(f"Quality Gate: {'PASSED ✅' if result['quality_gate_passed'] else 'FAILED ❌'}")
    print(f"\nTo view experiments: mlflow ui --port 5000")
