# ============================================================
# Dockerfile for Churn Prediction API
# ============================================================
# MLOps Concept: Containerization ensures the model runs
# identically in development, staging, and production.
# "Works on my machine" is eliminated with Docker.
#
# Multi-stage build:
#   Stage 1 (builder): Install dependencies into /install
#   Stage 2 (runtime): Copy /install to system Python path
# ============================================================

# ── Stage 1: Builder ────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /app

# Install build dependencies (needed to compile C extensions)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies to a dedicated prefix
# Using --prefix=/install so we can copy cleanly to Stage 2
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Runtime ────────────────────────────────────────
FROM python:3.10-slim AS runtime

# Security: Run as non-root user
RUN groupadd -r mlops && useradd -r -g mlops mlops

WORKDIR /app

# Copy installed packages from builder to system Python path
# This makes packages available to ALL users including mlops
COPY --from=builder /install /usr/local

# Copy application code
COPY src/ ./src/
COPY api/ ./api/
COPY configs/ ./configs/
COPY monitoring/ ./monitoring/
COPY pipelines/ ./pipelines/

# Create directories for artifacts and set ownership
RUN mkdir -p models reports mlruns mlartifacts data/raw data/processed && \
    chown -R mlops:mlops /app

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MLFLOW_TRACKING_URI=sqlite:///mlflow.db

# Expose API port
EXPOSE 8000

# Health check — Kubernetes uses this for liveness probe
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Switch to non-root user (security best practice)
USER mlops

# Start the FastAPI server
CMD ["uvicorn", "api.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info"]
