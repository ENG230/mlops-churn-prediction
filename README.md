# 🚀 MLOps End-to-End Demo: Customer Churn Prediction

> **A production-grade MLOps project demonstrating the complete ML lifecycle — from data ingestion to model monitoring — built for interview evaluation.**

[![CI/CD Pipeline](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-blue)](/.github/workflows/ci_cd.yml)
[![MLflow](https://img.shields.io/badge/Tracking-MLflow-orange)](http://localhost:5000)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-green)](http://localhost:8000/docs)
[![Docker](https://img.shields.io/badge/Container-Docker-blue)](Dockerfile)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow)](requirements.txt)

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Architecture](#-architecture)
3. [MLOps Components](#-mlops-components)
4. [Project Structure](#-project-structure)
5. [Quick Start](#-quick-start)
6. [Running Each Component](#-running-each-component)
7. [Tech Stack](#-tech-stack)
8. [Interview Q&A Guide](#-interview-qa-guide)

---

## 🎯 Project Overview

**Use Case:** Predict which telecom customers are likely to churn (cancel their subscription).

**Business Value:** Identifying at-risk customers allows the business to proactively offer retention incentives, reducing revenue loss.

**MLOps Value:** This project demonstrates how to take a model from a Jupyter notebook to a production-grade, monitored, continuously-improving ML system.

### Key Metrics Achieved
| Metric | Value |
|--------|-------|
| Accuracy | ~82% |
| ROC-AUC | ~88% |
| F1 Score | ~72% |
| API Latency | <50ms |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     MLOps Pipeline Architecture                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │   Data   │───▶│ Feature  │───▶│  Model   │───▶│  Model   │  │
│  │Ingestion │    │Engineer. │    │Training  │    │Registry  │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│       │               │               │                │         │
│  Validate         Preprocess      MLflow Run       Staging/     │
│  Schema           & Split         Tracking         Production   │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    CI/CD (GitHub Actions)                  │   │
│  │  Code Quality → Tests → Train → Quality Gate → Deploy     │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │  FastAPI     │    │   MLflow     │    │  Drift Detection  │  │
│  │  REST API    │    │   UI :5000   │    │  (PSI + Reports)  │  │
│  │  :8000       │    │              │    │                   │  │
│  └──────────────┘    └──────────────┘    └──────────────────┘  │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow
```
Raw Data (CSV)
    │
    ▼
Data Validation ──── FAIL ──▶ Alert & Stop
    │ PASS
    ▼
Feature Engineering (sklearn Pipeline)
    │
    ├── Numerical: Impute → StandardScaler
    └── Categorical: Impute → OneHotEncoder
    │
    ▼
Model Training (Random Forest / XGBoost)
    │
    ├── MLflow: Log params, metrics, artifacts
    └── Quality Gate: accuracy≥0.75, F1≥0.65, AUC≥0.80
    │
    ▼
Model Registry (MLflow)
    │
    ├── PASS → Staging → (Manual Review) → Production
    └── FAIL → Rejected (logged but not deployed)
    │
    ▼
FastAPI Serving (/predict, /predict/batch)
    │
    ▼
Monitoring (PSI Drift Detection → HTML Reports)
    │
    └── Drift Detected → Trigger Retraining
```

---

## 🔧 MLOps Components

### 1. 📊 Data Management
| Component | Implementation | File |
|-----------|---------------|------|
| Data Generation | Synthetic Telco Churn dataset | `src/data_ingestion.py` |
| Schema Validation | Custom validation with 5 checks | `src/data_ingestion.py` |
| Data Versioning | DVC pipeline stages | `dvc.yaml` |
| Train/Test Split | Stratified split (80/20) | `src/feature_engineering.py` |
| Reference Dataset | Saved for drift baseline | `data/processed/reference.csv` |

### 2. 🔬 Experiment Tracking (MLflow)
Every training run logs:
- **Parameters**: algorithm, n_estimators, max_depth, test_size, random_state
- **Metrics**: accuracy, precision, recall, F1, ROC-AUC (train + test)
- **Artifacts**: model, preprocessor, confusion matrix, ROC curve, feature importance
- **Tags**: quality_gate status, model_status

### 3. 🏭 Pipeline Orchestration
```
DVC Pipeline (dvc.yaml):
  data_ingestion → feature_engineering → train → drift_detection

Training Pipeline (pipelines/training_pipeline.py):
  Step 1: Data Ingestion
  Step 2: Feature Engineering  
  Step 3: Model Training
  Step 4: Model Registration
```

### 4. 🚦 Quality Gate
Before any model is promoted, it must pass:
```yaml
min_accuracy_threshold: 0.75
min_f1_threshold: 0.65
min_roc_auc_threshold: 0.80
```
**If quality gate fails → model is logged but NOT promoted → CI/CD pipeline fails**

### 5. 📦 Model Registry
MLflow Model Registry manages the model lifecycle:
```
None (just registered)
  ↓
Staging (quality gate passed, awaiting review)
  ↓
Production (manually approved, serving traffic)
  ↓
Archived (replaced by newer version)
```

### 6. 🌐 Model Serving (FastAPI)
```
GET  /health          → Liveness probe (Kubernetes)
GET  /ready           → Readiness probe (model loaded?)
GET  /info            → Model metadata & metrics
POST /predict         → Single customer prediction
POST /predict/batch   → Batch predictions (up to 1000)
GET  /metrics         → Operational metrics (Prometheus-ready)
GET  /docs            → Interactive Swagger UI
```

### 7. 🔍 Drift Detection
Monitors for distribution shift using **Population Stability Index (PSI)**:
```
PSI < 0.1:  ✅ STABLE    - No action needed
PSI 0.1-0.2: ⚠️ MODERATE  - Monitor closely
PSI > 0.2:  🚨 SIGNIFICANT - Investigate & retrain
```
Generates HTML reports with feature-level drift analysis.

### 8. 🔄 CI/CD Pipeline (GitHub Actions)
```yaml
Trigger: push to main, PR, weekly schedule, manual

Jobs:
  1. code-quality      → flake8, black, isort
  2. test              → pytest (data + model + API tests)
  3. train-and-evaluate → full pipeline + quality gate check
  4. drift-detection   → run drift analysis
  5. build-docker      → build & test Docker image
  6. deploy-staging    → deploy to staging environment
```

---

## 📁 Project Structure

```
mlops-churn-prediction/
│
├── 📂 src/                          # Core ML modules
│   ├── __init__.py
│   ├── data_ingestion.py            # Data loading & validation
│   ├── feature_engineering.py       # Preprocessing pipeline
│   ├── train.py                     # Training + MLflow tracking
│   └── predict.py                   # Inference module
│
├── 📂 pipelines/                    # Pipeline orchestration
│   └── training_pipeline.py         # End-to-end training pipeline
│
├── 📂 api/                          # Model serving
│   └── app.py                       # FastAPI application
│
├── 📂 monitoring/                   # Model monitoring
│   └── drift_detection.py           # PSI drift detection
│
├── 📂 tests/                        # Test suite
│   ├── test_data.py                 # Data validation tests
│   ├── test_model.py                # Model quality tests
│   └── test_api.py                  # API endpoint tests
│
├── 📂 configs/                      # Configuration
│   └── params.yaml                  # All hyperparameters (versioned)
│
├── 📂 .github/workflows/            # CI/CD
│   └── ci_cd.yml                    # GitHub Actions pipeline
│
├── 📂 data/                         # Data (DVC tracked)
│   ├── raw/                         # Raw data
│   └── processed/                   # Processed train/test/reference
│
├── 📂 models/                       # Saved artifacts
│   └── preprocessor.joblib          # Fitted preprocessor
│
├── 📂 reports/                      # Generated reports
│   ├── metrics.json                 # Model metrics
│   ├── confusion_matrix.png         # Confusion matrix plot
│   ├── roc_curve.png                # ROC curve plot
│   ├── feature_importance.png       # Feature importance plot
│   └── drift_report_*.html          # Drift detection reports
│
├── 📂 mlruns/                       # MLflow tracking data
│
├── Dockerfile                       # Container definition
├── docker-compose.yml               # Local stack orchestration
├── dvc.yaml                         # DVC pipeline definition
├── setup.py                         # Package setup
├── requirements.txt                 # Dependencies
└── README.md                        # This file
```

---

## ⚡ Quick Start

### Prerequisites
- Python 3.10+
- pip

### Option A: Run Locally (Recommended for Demo)

```bash
# 1. Clone / navigate to project
cd mlops-churn-prediction

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install project as package
pip install -e .

# 5. Run the full training pipeline
python pipelines/training_pipeline.py

# 6. Start MLflow UI (in a new terminal)
mlflow ui --port 5000
# Open: http://localhost:5000

# 7. Start the API server (in a new terminal)
uvicorn api.app:app --reload --port 8000
# Open: http://localhost:8000/docs

# 8. Run drift detection
python monitoring/drift_detection.py
# Open: reports/drift_report_*.html

# 9. Run all tests
pytest tests/ -v
```

### Option B: Run with Docker

```bash
# 1. Run training pipeline first (to generate model artifacts)
pip install -r requirements.txt
python pipelines/training_pipeline.py

# 2. Start full stack with Docker Compose
docker-compose up

# Services:
#   API:    http://localhost:8000/docs
#   MLflow: http://localhost:5000
```

---

## 🎮 Running Each Component

### Step 1: Data Ingestion
```bash
python src/data_ingestion.py
# Output: data/raw/churn_data.csv (7,043 synthetic records)
# Validates: schema, nulls, target values, row count
```

### Step 2: Feature Engineering
```bash
python src/feature_engineering.py
# Output: data/processed/train.csv, test.csv, reference.csv
# Output: models/preprocessor.joblib
```

### Step 3: Train Model
```bash
python src/train.py
# Output: MLflow run with all metrics and artifacts
# Output: reports/metrics.json, confusion_matrix.png, roc_curve.png
```

### Step 4: View Experiments in MLflow
```bash
mlflow ui --port 5000
# Navigate to: http://localhost:5000
# See: all runs, compare metrics, view artifacts
```

### Step 5: Run Full Pipeline
```bash
python pipelines/training_pipeline.py
# Runs all 4 steps in sequence with logging
```

### Step 6: Start API
```bash
uvicorn api.app:app --reload --port 8000
# Swagger UI: http://localhost:8000/docs
```

### Step 7: Make a Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
    "TotalCharges": 1026.0
  }'
```

### Step 8: Run Tests
```bash
# All tests
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=src --cov-report=html

# Specific test file
pytest tests/test_data.py -v
pytest tests/test_model.py -v
pytest tests/test_api.py -v
```

### Step 9: Run Drift Detection
```bash
# Simulate drift (for demo)
python monitoring/drift_detection.py --drift-factor 0.4

# Use actual test data (no simulation)
python monitoring/drift_detection.py --no-simulate

# Open HTML report
# reports/drift_report_*.html
```

### Step 10: Change Algorithm (Config-Driven)
```bash
# Edit configs/params.yaml:
#   model:
#     algorithm: "xgboost"   # Change from random_forest

# Re-run pipeline - everything updates automatically
python pipelines/training_pipeline.py
```

---

## 🛠️ Tech Stack

| Category | Tool | Purpose |
|----------|------|---------|
| **Language** | Python 3.10 | Core language |
| **ML Framework** | scikit-learn, XGBoost | Model training |
| **Experiment Tracking** | MLflow | Log params, metrics, artifacts |
| **Model Registry** | MLflow Model Registry | Model versioning & lifecycle |
| **Pipeline** | DVC | Reproducible ML pipelines |
| **API Serving** | FastAPI + Uvicorn | REST API for predictions |
| **Validation** | Pydantic v2 | Input schema validation |
| **Containerization** | Docker + docker-compose | Reproducible environments |
| **CI/CD** | GitHub Actions | Automated test/train/deploy |
| **Monitoring** | Custom PSI + HTML reports | Drift detection |
| **Testing** | pytest | Unit, data, API tests |
| **Config** | YAML (params.yaml) | Externalized hyperparameters |

---

## 🎤 Interview Q&A Guide

### Q1: "Walk me through your MLOps pipeline"
> **Answer:** "The pipeline has 4 stages orchestrated by DVC and the training pipeline script:
> 1. **Data Ingestion** — loads data, validates schema (5 checks), ensures quality
> 2. **Feature Engineering** — sklearn ColumnTransformer pipeline (imputation + scaling + encoding), fitted on train, applied to test
> 3. **Model Training** — trains Random Forest/XGBoost, logs everything to MLflow (params, metrics, plots, model artifact)
> 4. **Model Registration** — if quality gate passes (accuracy≥0.75, F1≥0.65, AUC≥0.80), model is promoted to Staging in MLflow Registry
> 
> The whole pipeline is triggered by GitHub Actions on every push to main."

---

### Q2: "How do you handle model versioning?"
> **Answer:** "I use MLflow Model Registry. Every training run registers the model with a version number. Models go through stages: None → Staging → Production → Archived. The API loads the model by name and stage (`models:/churn-classifier/Production`), not by file path. This decouples the serving layer from training — I can retrain and promote a new version without changing the API code."

---

### Q3: "How do you detect model degradation in production?"
> **Answer:** "I use Population Stability Index (PSI) for numerical features and distribution shift analysis for categorical features. PSI < 0.1 is stable, 0.1-0.2 is moderate, >0.2 is significant. I compare production data against the reference dataset (training data). The monitoring script generates HTML reports and logs JSON results. If significant drift is detected, it triggers a retraining recommendation. In production, this would be scheduled daily and alert via Slack/PagerDuty."

---

### Q4: "How do you ensure reproducibility?"
> **Answer:** "Three layers:
> 1. **Data**: DVC tracks data files with checksums — same data hash = same data
> 2. **Code**: Git tracks all code changes
> 3. **Config**: All hyperparameters are in `params.yaml`, never hardcoded. DVC tracks param changes and knows which stages need to re-run
> 
> The sklearn Pipeline ensures the exact same preprocessing is applied at training and inference — the fitted preprocessor is saved as a joblib artifact and loaded at serving time."

---

### Q5: "How does your CI/CD pipeline work for ML?"
> **Answer:** "GitHub Actions runs 6 jobs on every push to main:
> 1. Code quality (flake8, black)
> 2. Tests (data tests, model tests, API schema tests)
> 3. Training pipeline + quality gate check (if gate fails, pipeline fails, no deployment)
> 4. Drift detection
> 5. Docker build and health check test
> 6. Deploy to staging
> 
> The key difference from regular CI/CD is the quality gate — if the model doesn't meet minimum performance thresholds, the pipeline fails and the model is not deployed."

---

### Q6: "How do you deploy models?"
> **Answer:** "The model is served via a FastAPI REST API containerized with Docker. The API has:
> - `/health` and `/ready` endpoints for Kubernetes liveness/readiness probes
> - `/predict` for single predictions with Pydantic input validation
> - `/predict/batch` for batch scoring
> - `/metrics` for Prometheus scraping
> 
> The model is loaded once at startup from MLflow Registry (not on every request). Docker Compose orchestrates the API + MLflow server locally. In production, this would be a Kubernetes deployment."

---

### Q7: "How do you manage configurations?"
> **Answer:** "All hyperparameters, file paths, thresholds, and feature lists are in `configs/params.yaml`. Nothing is hardcoded. To change the algorithm from Random Forest to XGBoost, I just change one line in params.yaml and re-run the pipeline. DVC tracks param changes and knows which stages need to re-run. This is the config-driven approach — the code is generic, the config drives behavior."

---

### Q8: "What's the difference between data drift and concept drift?"
> **Answer:** "Data drift (covariate shift) is when the input feature distributions change — e.g., more customers with short tenure, higher monthly charges. The model's learned relationships may still be valid, but it's seeing inputs it wasn't trained on. Concept drift is when the relationship between features and the target changes — e.g., customers with fiber optic internet used to churn more, but now they don't. This is harder to detect and requires labeled production data. I detect data drift with PSI. For concept drift, you need to monitor prediction accuracy on labeled production data over time."

---

### Q9: "How do you handle the training-serving skew problem?"
> **Answer:** "Training-serving skew happens when preprocessing at training time differs from preprocessing at inference time. I solve this by saving the fitted sklearn ColumnTransformer as a joblib artifact during training. At inference, the API loads this exact same fitted preprocessor. The same imputation values, scaling parameters, and one-hot encoding categories are used. This is enforced by the `engineer_features()` function which takes a `fit=True/False` parameter."

---

### Q10: "What would you add to make this production-ready?"
> **Answer:** "Several things:
> 1. **Remote storage**: DVC remote (S3/GCS) for data and model artifacts
> 2. **Feature Store**: Feast or Tecton for feature reuse across models
> 3. **A/B Testing**: Shadow deployment to compare champion vs challenger model
> 4. **Prometheus + Grafana**: Replace custom metrics endpoint with proper observability stack
> 5. **Kubernetes**: Replace docker-compose with K8s manifests + HPA for auto-scaling
> 6. **Data lineage**: Track which data version trained which model version
> 7. **Model explainability**: SHAP values for individual predictions
> 8. **Retraining automation**: Trigger retraining automatically when drift exceeds threshold"

---

## 📊 MLOps Maturity Assessment

This project demonstrates **Level 2 MLOps**:

| Level | Description | This Project |
|-------|-------------|-------------|
| Level 0 | Manual, notebook-based | ❌ |
| Level 1 | Automated training pipeline | ✅ |
| Level 2 | CI/CD for ML | ✅ |
| Level 3 | Full MLOps (feature store, A/B, auto-retrain) | 🔄 Partial |

---

## 📄 License

MIT License — Free to use for learning and interviews.

---

*Built to demonstrate production MLOps practices. Every design decision has a reason — ask about any of them in your interview!*
