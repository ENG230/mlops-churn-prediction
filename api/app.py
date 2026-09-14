"""
FastAPI Model Serving Application
===================================
REST API for serving the churn prediction model.

MLOps Concepts Demonstrated:
- Model serving as a microservice
- Health check endpoint (for Kubernetes liveness/readiness probes)
- Input validation with Pydantic schemas
- Prediction with confidence scores
- Metrics endpoint for monitoring
- Lazy model loading (load once, serve many)
- Structured logging for observability

Endpoints:
  GET  /health       → Health check (liveness probe)
  GET  /ready        → Readiness check (model loaded?)
  GET  /info         → Model info and metadata
  POST /predict      → Single prediction
  POST /predict/batch → Batch predictions
  GET  /metrics      → Prediction statistics
"""

import sys
import os
import logging
import time
import json
from pathlib import Path
from typing import Optional, List
from datetime import datetime
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import pandas as pd
import numpy as np
import mlflow
import joblib
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Pydantic Schemas (Input Validation)
# ─────────────────────────────────────────────────────────────

class CustomerFeatures(BaseModel):
    """
    Input schema for a single customer prediction request.
    Pydantic validates all fields automatically.
    """
    gender: str = Field(..., example="Male", description="Customer gender: Male or Female")
    SeniorCitizen: int = Field(..., ge=0, le=1, example=0, description="1 if senior citizen, 0 otherwise")
    Partner: str = Field(..., example="Yes", description="Has partner: Yes or No")
    Dependents: str = Field(..., example="No", description="Has dependents: Yes or No")
    tenure: int = Field(..., ge=0, le=100, example=12, description="Months with company")
    PhoneService: str = Field(..., example="Yes")
    MultipleLines: str = Field(..., example="No")
    InternetService: str = Field(..., example="Fiber optic")
    OnlineSecurity: str = Field(..., example="No")
    OnlineBackup: str = Field(..., example="No")
    DeviceProtection: str = Field(..., example="No")
    TechSupport: str = Field(..., example="No")
    StreamingTV: str = Field(..., example="Yes")
    StreamingMovies: str = Field(..., example="Yes")
    Contract: str = Field(..., example="Month-to-month")
    PaperlessBilling: str = Field(..., example="Yes")
    PaymentMethod: str = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., ge=0, le=200, example=85.50)
    TotalCharges: float = Field(..., ge=0, example=1026.0)

    class Config:
        json_schema_extra = {
            "example": {
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
            }
        }


class BatchPredictionRequest(BaseModel):
    """Input schema for batch prediction."""
    customers: List[CustomerFeatures] = Field(..., min_length=1, max_length=1000)


class PredictionResponse(BaseModel):
    """Output schema for a single prediction."""
    prediction: int
    prediction_label: str
    churn_probability: float
    no_churn_probability: float
    confidence: float
    model_version: str
    prediction_id: str
    timestamp: str


class BatchPredictionResponse(BaseModel):
    """Output schema for batch predictions."""
    predictions: List[dict]
    total_records: int
    churn_count: int
    churn_rate: float
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    uptime_seconds: float


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    algorithm: str
    status: str
    features_count: int
    training_metrics: dict


# ─────────────────────────────────────────────────────────────
# Application State (Model Cache)
# ─────────────────────────────────────────────────────────────

class ModelState:
    """
    Singleton to hold loaded model and preprocessor.
    MLOps Concept: Load model once at startup, not on every request.
    """
    model = None
    preprocessor = None
    config = None
    model_version = "unknown"
    algorithm = "unknown"
    is_ready = False
    start_time = time.time()
    prediction_count = 0
    churn_predictions = 0
    prediction_latencies = []


app_state = ModelState()


# ─────────────────────────────────────────────────────────────
# FastAPI Application
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Churn Prediction API",
    description="""
## MLOps Churn Prediction Service

This API serves a machine learning model that predicts customer churn.

### MLOps Features:
- **Model Registry Integration**: Model loaded from MLflow Model Registry
- **Health Checks**: Liveness and readiness probes for Kubernetes
- **Input Validation**: Pydantic schema validation on all inputs
- **Observability**: Structured logging and metrics endpoint
- **Versioning**: Model version tracked in every response

### Usage:
1. Check health: `GET /health`
2. Single prediction: `POST /predict`
3. Batch prediction: `POST /predict/batch`
4. View metrics: `GET /metrics`
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────
# Startup / Shutdown Events
# ─────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """
    Load model and preprocessor at startup.
    MLOps Concept: Eager loading prevents cold-start latency on first request.
    """
    logger.info("🚀 Starting Churn Prediction API...")
    load_model_and_preprocessor()


def load_model_and_preprocessor(config_path: str = "configs/params.yaml"):
    """Load model from MLflow registry or fallback to file."""
    try:
        with open(config_path, "r") as f:
            app_state.config = yaml.safe_load(f)

        tracking_uri = app_state.config["mlflow"]["tracking_uri"]
        model_name = app_state.config["mlflow"]["model_name"]

        mlflow.set_tracking_uri(tracking_uri)

        # Try loading from registry (Production stage first)
        for stage in ["Production", "Staging", "None"]:
            try:
                model_uri = f"models:/{model_name}/{stage}"
                app_state.model = mlflow.sklearn.load_model(model_uri)
                app_state.model_version = f"{stage}"
                logger.info(f"✅ Model loaded from registry: {model_uri}")
                break
            except Exception:
                continue

        # Fallback: load from latest MLflow run
        if app_state.model is None:
            try:
                client = mlflow.tracking.MlflowClient()
                experiment = client.get_experiment_by_name(
                    app_state.config["mlflow"]["experiment_name"]
                )
                if experiment:
                    runs = client.search_runs(
                        experiment_ids=[experiment.experiment_id],
                        order_by=["start_time DESC"],
                        max_results=1
                    )
                    if runs:
                        run_id = runs[0].info.run_id
                        model_uri = f"runs:/{run_id}/model"
                        app_state.model = mlflow.sklearn.load_model(model_uri)
                        app_state.model_version = f"run/{run_id[:8]}"
                        logger.info(f"✅ Model loaded from latest run: {run_id[:8]}")
            except Exception as e:
                logger.warning(f"Could not load from MLflow runs: {e}")

        # Try loading model directly from joblib file first (most reliable for Docker)
        model_joblib_path = "models/model.joblib"
        if Path(model_joblib_path).exists():
            app_state.model = joblib.load(model_joblib_path)
            app_state.model_version = "joblib/latest"
            logger.info(f"✅ Model loaded from file: {model_joblib_path}")

        # Load preprocessor
        preprocessor_path = "models/preprocessor.joblib"
        if Path(preprocessor_path).exists():
            app_state.preprocessor = joblib.load(preprocessor_path)
            logger.info(f"✅ Preprocessor loaded from {preprocessor_path}")
        else:
            logger.warning(f"⚠️  Preprocessor not found at {preprocessor_path}")

        # Determine algorithm
        if app_state.model is not None:
            app_state.algorithm = type(app_state.model).__name__
            app_state.is_ready = app_state.preprocessor is not None
            logger.info(f"✅ API ready! Algorithm: {app_state.algorithm}")
        else:
            logger.warning("⚠️  No model loaded. Run training pipeline first.")
            app_state.is_ready = False

    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        app_state.is_ready = False


# ─────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────

def make_prediction(customer_data: dict) -> dict:
    """Core prediction logic."""
    from src.feature_engineering import preprocess_raw_data, engineer_features

    config = app_state.config
    df = pd.DataFrame([customer_data])

    # Add dummy target for preprocessing
    target = config["data"]["target_column"]
    df[target] = 0

    # Preprocess
    df = preprocess_raw_data(df, config)

    # Transform
    X, _, _, _ = engineer_features(
        df, config,
        preprocessor=app_state.preprocessor,
        fit=False,
        save_preprocessor=False
    )

    # Predict
    prediction = int(app_state.model.predict(X)[0])
    probabilities = app_state.model.predict_proba(X)[0]

    return {
        "prediction": prediction,
        "prediction_label": "Churn" if prediction == 1 else "No Churn",
        "churn_probability": round(float(probabilities[1]), 4),
        "no_churn_probability": round(float(probabilities[0]), 4),
        "confidence": round(float(max(probabilities)), 4),
    }


# ─────────────────────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Liveness probe endpoint.
    Returns 200 if the service is running.
    Used by Kubernetes/Docker to check if container is alive.
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        uptime_seconds=round(time.time() - app_state.start_time, 2)
    )


@app.get("/ready", tags=["Health"])
async def readiness_check():
    """
    Readiness probe endpoint.
    Returns 200 only if model is loaded and ready to serve.
    Used by Kubernetes to decide if traffic should be routed here.
    """
    if app_state.is_ready:
        return {
            "status": "ready",
            "model_loaded": True,
            "preprocessor_loaded": app_state.preprocessor is not None,
            "timestamp": datetime.utcnow().isoformat()
        }
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Model not loaded. Run training pipeline first."
    )


@app.get("/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    """Get model metadata and training metrics."""
    if not app_state.is_ready:
        raise HTTPException(status_code=503, detail="Model not ready")

    # Load metrics if available
    metrics = {}
    metrics_path = "reports/metrics.json"
    if Path(metrics_path).exists():
        with open(metrics_path) as f:
            data = json.load(f)
            metrics = data.get("test", {})

    n_features = 0
    if app_state.preprocessor is not None:
        try:
            n_features = len(app_state.preprocessor.get_feature_names_out())
        except Exception:
            pass

    return ModelInfoResponse(
        model_name=app_state.config["mlflow"]["model_name"],
        model_version=app_state.model_version,
        algorithm=app_state.algorithm,
        status="ready" if app_state.is_ready else "not_ready",
        features_count=n_features,
        training_metrics=metrics
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_single(customer: CustomerFeatures, request: Request):
    """
    Single customer churn prediction.
    
    Returns prediction (0/1), label, and probability scores.
    """
    if not app_state.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not ready. Run training pipeline first."
        )

    start_time = time.time()

    try:
        customer_dict = customer.model_dump()
        result = make_prediction(customer_dict)

        # Update metrics
        app_state.prediction_count += 1
        if result["prediction"] == 1:
            app_state.churn_predictions += 1
        latency = time.time() - start_time
        app_state.prediction_latencies.append(latency)

        # Keep only last 1000 latencies
        if len(app_state.prediction_latencies) > 1000:
            app_state.prediction_latencies = app_state.prediction_latencies[-1000:]

        prediction_id = f"pred_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"

        logger.info(
            f"Prediction: {result['prediction_label']} | "
            f"Churn prob: {result['churn_probability']:.3f} | "
            f"Latency: {latency*1000:.1f}ms"
        )

        return PredictionResponse(
            **result,
            model_version=app_state.model_version,
            prediction_id=prediction_id,
            timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(batch_request: BatchPredictionRequest):
    """
    Batch prediction for multiple customers.
    
    Accepts up to 1000 customers per request.
    """
    if not app_state.is_ready:
        raise HTTPException(status_code=503, detail="Model not ready")

    start_time = time.time()
    predictions = []

    try:
        for i, customer in enumerate(batch_request.customers):
            customer_dict = customer.model_dump()
            result = make_prediction(customer_dict)
            result["customer_index"] = i
            predictions.append(result)

        churn_count = sum(1 for p in predictions if p["prediction"] == 1)
        churn_rate = churn_count / len(predictions) if predictions else 0

        app_state.prediction_count += len(predictions)
        app_state.churn_predictions += churn_count

        latency = time.time() - start_time
        logger.info(
            f"Batch prediction: {len(predictions)} records | "
            f"Churn rate: {churn_rate:.2%} | "
            f"Latency: {latency*1000:.1f}ms"
        )

        return BatchPredictionResponse(
            predictions=predictions,
            total_records=len(predictions),
            churn_count=churn_count,
            churn_rate=round(churn_rate, 4),
            timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """
    Prediction metrics for monitoring.
    
    MLOps Concept: Expose operational metrics for monitoring dashboards.
    In production, these would be scraped by Prometheus.
    """
    avg_latency = (
        sum(app_state.prediction_latencies) / len(app_state.prediction_latencies)
        if app_state.prediction_latencies else 0
    )
    churn_rate = (
        app_state.churn_predictions / app_state.prediction_count
        if app_state.prediction_count > 0 else 0
    )

    return {
        "total_predictions": app_state.prediction_count,
        "churn_predictions": app_state.churn_predictions,
        "no_churn_predictions": app_state.prediction_count - app_state.churn_predictions,
        "churn_rate": round(churn_rate, 4),
        "avg_latency_ms": round(avg_latency * 1000, 2),
        "model_version": app_state.model_version,
        "algorithm": app_state.algorithm,
        "uptime_seconds": round(time.time() - app_state.start_time, 2),
        "is_ready": app_state.is_ready,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/", tags=["Root"])
async def root():
    """API root - returns basic info."""
    return {
        "service": "Churn Prediction API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
