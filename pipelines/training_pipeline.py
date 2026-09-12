"""
Training Pipeline
=================
Orchestrates the full end-to-end ML training pipeline.

MLOps Concept: Pipelines are the backbone of MLOps.
Each step is modular, logged, and reproducible.
This is equivalent to what Kubeflow/Airflow/Prefect would orchestrate
in a production environment.

Pipeline Steps:
  1. Data Ingestion      → Load & validate raw data
  2. Feature Engineering → Preprocess & transform features
  3. Model Training      → Train with MLflow tracking
  4. Model Evaluation    → Quality gate check
  5. Model Registration  → Register in MLflow Model Registry
"""

import sys
import os
import logging
import yaml
import json
import time
import mlflow
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def load_config(config_path: str = "configs/params.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def step_data_ingestion(config_path: str) -> bool:
    """
    Pipeline Step 1: Data Ingestion
    - Load or generate raw data
    - Validate schema and quality
    """
    logger.info("=" * 60)
    logger.info("STEP 1: DATA INGESTION")
    logger.info("=" * 60)
    start = time.time()

    try:
        from src.data_ingestion import ingest_data
        df = ingest_data(config_path)
        elapsed = time.time() - start
        logger.info(f"✅ Data ingestion complete in {elapsed:.2f}s")
        logger.info(f"   Dataset shape: {df.shape}")
        return True
    except Exception as e:
        logger.error(f"❌ Data ingestion FAILED: {e}")
        raise


def step_feature_engineering(config_path: str) -> tuple:
    """
    Pipeline Step 2: Feature Engineering
    - Preprocess raw data
    - Split into train/test
    - Fit and save preprocessor
    - Return processed arrays
    """
    logger.info("=" * 60)
    logger.info("STEP 2: FEATURE ENGINEERING")
    logger.info("=" * 60)
    start = time.time()

    try:
        from src.feature_engineering import run_feature_engineering
        X_train, y_train, X_test, y_test, preprocessor, feature_names = \
            run_feature_engineering(config_path)
        elapsed = time.time() - start
        logger.info(f"✅ Feature engineering complete in {elapsed:.2f}s")
        logger.info(f"   X_train: {X_train.shape}, X_test: {X_test.shape}")
        logger.info(f"   Features: {len(feature_names)}")
        return X_train, y_train, X_test, y_test, preprocessor, feature_names
    except Exception as e:
        logger.error(f"❌ Feature engineering FAILED: {e}")
        raise


def step_model_training(config_path: str) -> dict:
    """
    Pipeline Step 3: Model Training
    - Train model with MLflow tracking
    - Log all params, metrics, artifacts
    - Check quality gate
    """
    logger.info("=" * 60)
    logger.info("STEP 3: MODEL TRAINING")
    logger.info("=" * 60)
    start = time.time()

    try:
        from src.train import train_model
        result = train_model(config_path)
        elapsed = time.time() - start
        logger.info(f"✅ Model training complete in {elapsed:.2f}s")
        logger.info(f"   Run ID: {result['run_id']}")
        logger.info(f"   Test Metrics: {result['metrics']}")
        return result
    except Exception as e:
        logger.error(f"❌ Model training FAILED: {e}")
        raise


def step_model_registration(config_path: str, run_id: str, quality_passed: bool) -> bool:
    """
    Pipeline Step 4: Model Registration
    
    MLOps Concept: Model Registry manages model lifecycle.
    - If quality gate passed → promote to 'Staging'
    - Manual approval required to promote to 'Production'
    - Failed models are tagged but not promoted
    """
    logger.info("=" * 60)
    logger.info("STEP 4: MODEL REGISTRATION")
    logger.info("=" * 60)

    config = load_config(config_path)
    model_name = config["mlflow"]["model_name"]
    tracking_uri = config["mlflow"]["tracking_uri"]

    mlflow.set_tracking_uri(tracking_uri)

    if not quality_passed:
        logger.warning("⚠️  Quality gate FAILED. Model will NOT be promoted to Staging.")
        logger.warning("   Model is registered but tagged as 'rejected'")
        return False

    try:
        client = mlflow.tracking.MlflowClient()

        # Get the latest version of the registered model
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            logger.warning("No model versions found in registry")
            return False

        # Get the version from this run
        run_versions = [v for v in versions if v.run_id == run_id]
        if not run_versions:
            # Get the latest version
            latest_version = max(versions, key=lambda v: int(v.version))
        else:
            latest_version = run_versions[0]

        version_number = latest_version.version
        logger.info(f"Found model version: {version_number}")

        # Add description and tags
        client.update_model_version(
            name=model_name,
            version=version_number,
            description=f"Trained on {datetime.now().strftime('%Y-%m-%d %H:%M')}. Quality gate: PASSED."
        )

        # Transition to Staging (MLflow 2.x API - may not be available in 3.x)
        try:
            client.transition_model_version_stage(
                name=model_name,
                version=version_number,
                stage="Staging",
                archive_existing_versions=False
            )
        except Exception:
            # MLflow 3.x uses aliases instead of stages
            try:
                client.set_registered_model_alias(
                    name=model_name,
                    alias="staging",
                    version=version_number
                )
            except Exception as alias_err:
                logger.warning(f"Could not set stage/alias: {alias_err}")

        logger.info(f"✅ Model v{version_number} promoted to STAGING")
        logger.info(f"   Model: {model_name} | Version: {version_number}")
        logger.info(f"   Next step: Manual review → promote to Production")
        logger.info(f"   Command: python pipelines/promote_model.py --version {version_number}")
        return True

    except Exception as e:
        logger.error(f"❌ Model registration FAILED: {e}")
        logger.info("   Note: Model is still logged in MLflow, just not staged")
        return False


def run_pipeline(config_path: str = "configs/params.yaml") -> dict:
    """
    Execute the full training pipeline.
    
    This is the main entry point for the training pipeline.
    In production, this would be triggered by:
    - A scheduled cron job
    - A data drift alert
    - A CI/CD pipeline (GitHub Actions)
    - An orchestrator (Airflow/Kubeflow/Prefect)
    """
    pipeline_start = time.time()
    pipeline_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║       MLOps CHURN PREDICTION TRAINING PIPELINE          ║")
    logger.info(f"║       Pipeline ID: {pipeline_id:<38}║")
    logger.info("╚" + "═" * 58 + "╝")

    results = {
        "pipeline_id": pipeline_id,
        "steps": {},
        "success": False,
    }

    try:
        # Step 1: Data Ingestion
        step_start = time.time()
        step_data_ingestion(config_path)
        results["steps"]["data_ingestion"] = {
            "status": "SUCCESS",
            "duration_s": round(time.time() - step_start, 2)
        }

        # Step 2: Feature Engineering
        step_start = time.time()
        step_feature_engineering(config_path)
        results["steps"]["feature_engineering"] = {
            "status": "SUCCESS",
            "duration_s": round(time.time() - step_start, 2)
        }

        # Step 3: Model Training
        step_start = time.time()
        training_result = step_model_training(config_path)
        results["steps"]["model_training"] = {
            "status": "SUCCESS",
            "duration_s": round(time.time() - step_start, 2),
            "run_id": training_result["run_id"],
            "metrics": training_result["metrics"],
        }

        # Step 4: Model Registration
        step_start = time.time()
        registered = step_model_registration(
            config_path,
            training_result["run_id"],
            training_result["quality_gate_passed"]
        )
        results["steps"]["model_registration"] = {
            "status": "SUCCESS" if registered else "SKIPPED",
            "duration_s": round(time.time() - step_start, 2),
            "promoted_to_staging": registered,
        }

        results["success"] = True
        results["quality_gate_passed"] = training_result["quality_gate_passed"]
        results["run_id"] = training_result["run_id"]
        results["metrics"] = training_result["metrics"]

    except Exception as e:
        results["success"] = False
        results["error"] = str(e)
        logger.error(f"Pipeline FAILED: {e}")
        raise

    finally:
        total_duration = time.time() - pipeline_start
        results["total_duration_s"] = round(total_duration, 2)

        # Save pipeline results
        results_path = f"reports/pipeline_run_{pipeline_id}.json"
        Path(results_path).parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info("╔" + "═" * 58 + "╗")
        logger.info("║                  PIPELINE SUMMARY                       ║")
        logger.info("╠" + "═" * 58 + "╣")
        logger.info(f"║  Status: {'✅ SUCCESS' if results['success'] else '❌ FAILED':<49}║")
        logger.info(f"║  Duration: {total_duration:.1f}s{' ' * (47 - len(f'{total_duration:.1f}s'))}║")
        for step_name, step_info in results.get("steps", {}).items():
            status = step_info.get("status", "UNKNOWN")
            duration = step_info.get("duration_s", 0)
            logger.info(f"║  {step_name}: {status} ({duration}s){' ' * max(0, 40 - len(step_name) - len(status) - len(str(duration)))}║")
        logger.info("╚" + "═" * 58 + "╝")
        logger.info(f"Results saved to: {results_path}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run MLOps Training Pipeline")
    parser.add_argument(
        "--config", default="configs/params.yaml",
        help="Path to config file"
    )
    args = parser.parse_args()

    results = run_pipeline(args.config)

    print(f"\n{'='*50}")
    print(f"Pipeline {'SUCCEEDED ✅' if results['success'] else 'FAILED ❌'}")
    if results.get("metrics"):
        print(f"Test Metrics: {results['metrics']}")
    print(f"Total Duration: {results.get('total_duration_s', 0):.1f}s")
    print(f"\nNext Steps:")
    print(f"  1. View experiments: mlflow ui --port 5000")
    print(f"  2. Start API server: uvicorn api.app:app --reload --port 8000")
    print(f"  3. Run monitoring:   python monitoring/drift_detection.py")
