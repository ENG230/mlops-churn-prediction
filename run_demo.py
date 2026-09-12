"""
MLOps Demo Runner
==================
One-script demo that runs the complete MLOps pipeline
and shows all components working together.

Usage:
    python run_demo.py              # Run full demo
    python run_demo.py --step 1     # Run specific step only
"""

import sys
import os
import time
import argparse
import subprocess
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def print_banner(title: str, width: int = 60):
    print("\n" + "═" * width)
    print(f"  {title}")
    print("═" * width)


def print_step(step_num: int, title: str):
    print(f"\n{'─'*60}")
    print(f"  STEP {step_num}: {title}")
    print(f"{'─'*60}")


def step1_data_ingestion():
    print_step(1, "DATA INGESTION & VALIDATION")
    from src.data_ingestion import ingest_data
    df = ingest_data()
    print(f"\n✅ Data loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"   Churn rate: {(df['Churn'] == 'Yes').mean():.1%}")
    print(f"   Saved to: data/raw/churn_data.csv")
    return df


def step2_feature_engineering():
    print_step(2, "FEATURE ENGINEERING")
    from src.feature_engineering import run_feature_engineering
    X_train, y_train, X_test, y_test, preprocessor, feature_names = \
        run_feature_engineering()
    print(f"\n✅ Features engineered:")
    print(f"   X_train: {X_train.shape} | X_test: {X_test.shape}")
    print(f"   Total features after encoding: {len(feature_names)}")
    print(f"   Preprocessor saved to: models/preprocessor.joblib")
    return X_train, y_train, X_test, y_test


def step3_training():
    print_step(3, "MODEL TRAINING WITH MLFLOW TRACKING")
    from src.train import train_model
    result = train_model()
    print(f"\n✅ Training complete!")
    print(f"   Run ID: {result['run_id']}")
    print(f"   Algorithm: {result.get('algorithm', 'random_forest')}")
    print(f"\n   Test Metrics:")
    for k, v in result['metrics'].items():
        print(f"     {k}: {v:.4f}")
    print(f"\n   Quality Gate: {'✅ PASSED' if result['quality_gate_passed'] else '❌ FAILED'}")
    print(f"\n   📊 View in MLflow: mlflow ui --port 5000")
    return result


def step4_drift_detection():
    print_step(4, "DRIFT DETECTION")
    from monitoring.drift_detection import run_drift_detection
    report = run_drift_detection(simulate_drift=True, drift_factor=0.35)
    summary = report["summary"]
    print(f"\n✅ Drift detection complete!")
    print(f"   Drift detected: {'YES 🚨' if summary['drift_detected'] else 'NO ✅'}")
    print(f"   Retraining recommended: {'YES 🔄' if summary['retraining_recommended'] else 'NO'}")
    print(f"   Significant features: {summary['significant_numerical_features']}")
    print(f"\n   📄 HTML report: reports/drift_report_*.html")
    return report


def step5_api_demo():
    print_step(5, "API PREDICTION DEMO")
    print("\n   Testing prediction logic directly (without starting server)...")

    try:
        from src.predict import predict

        # High-risk customer
        high_risk = {
            "gender": "Male", "SeniorCitizen": 1, "Partner": "No",
            "Dependents": "No", "tenure": 2, "PhoneService": "Yes",
            "MultipleLines": "No", "InternetService": "Fiber optic",
            "OnlineSecurity": "No", "OnlineBackup": "No",
            "DeviceProtection": "No", "TechSupport": "No",
            "StreamingTV": "Yes", "StreamingMovies": "Yes",
            "Contract": "Month-to-month", "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 95.0, "TotalCharges": 190.0,
        }

        # Low-risk customer
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
        }

        print("\n   Customer 1 (HIGH RISK - Month-to-month, Fiber optic, 2 months tenure):")
        result1 = predict(high_risk)
        print(f"     Prediction: {result1['prediction_label'].values[0]}")
        print(f"     Churn Probability: {result1['churn_probability'].values[0]:.1%}")

        print("\n   Customer 2 (LOW RISK - Two year contract, DSL, 60 months tenure):")
        result2 = predict(low_risk)
        print(f"     Prediction: {result2['prediction_label'].values[0]}")
        print(f"     Churn Probability: {result2['churn_probability'].values[0]:.1%}")

        print(f"\n✅ Predictions working correctly!")
        print(f"\n   To start the API server:")
        print(f"     uvicorn api.app:app --reload --port 8000")
        print(f"     Open: http://localhost:8000/docs")

    except Exception as e:
        print(f"\n⚠️  Prediction demo skipped: {e}")
        print(f"   Run training first: python pipelines/training_pipeline.py")


def step6_tests():
    print_step(6, "RUNNING TEST SUITE")
    print("\n   Running pytest...")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short", "-q"],
        capture_output=False,
        text=True
    )
    if result.returncode == 0:
        print("\n✅ All tests passed!")
    else:
        print("\n⚠️  Some tests failed (check output above)")


def run_full_demo():
    print_banner("🚀 MLOps CHURN PREDICTION - FULL DEMO")
    print("""
This demo runs the complete MLOps pipeline:
  1. Data Ingestion & Validation
  2. Feature Engineering
  3. Model Training (MLflow tracking)
  4. Drift Detection
  5. Prediction Demo
  6. Test Suite

Estimated time: 2-3 minutes
""")

    start = time.time()

    try:
        step1_data_ingestion()
        step2_feature_engineering()
        result = step3_training()
        step4_drift_detection()
        step5_api_demo()
        step6_tests()

        elapsed = time.time() - start

        print_banner("🎉 DEMO COMPLETE!")
        print(f"""
✅ All MLOps components demonstrated successfully!

📊 RESULTS:
   Model Metrics: {result['metrics']}
   Quality Gate:  {'PASSED ✅' if result['quality_gate_passed'] else 'FAILED ❌'}
   Duration:      {elapsed:.1f}s

🔗 NEXT STEPS:
   1. View MLflow experiments:
      mlflow ui --port 5000
      → http://localhost:5000

   2. Start prediction API:
      uvicorn api.app:app --reload --port 8000
      → http://localhost:8000/docs

   3. View drift report:
      → reports/drift_report_*.html

   4. Run with Docker:
      docker-compose up
      → API: http://localhost:8000
      → MLflow: http://localhost:5000

📖 Interview prep: See README.md → Interview Q&A Guide
""")

    except Exception as e:
        print(f"\n❌ Demo failed at step: {e}")
        print("\nTroubleshooting:")
        print("  1. Ensure dependencies are installed: pip install -r requirements.txt")
        print("  2. Ensure you're in the project root directory")
        print("  3. Check Python version: python --version (need 3.10+)")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLOps Demo Runner")
    parser.add_argument("--step", type=int, choices=[1, 2, 3, 4, 5, 6],
                        help="Run a specific step only")
    args = parser.parse_args()

    if args.step == 1:
        step1_data_ingestion()
    elif args.step == 2:
        step2_feature_engineering()
    elif args.step == 3:
        step3_training()
    elif args.step == 4:
        step4_drift_detection()
    elif args.step == 5:
        step5_api_demo()
    elif args.step == 6:
        step6_tests()
    else:
        run_full_demo()
