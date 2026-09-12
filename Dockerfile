# ============================================================
# Dockerfile for Churn Prediction API
# ============================================================
# MLOps Concept: Containerization ensures the model runs
# identically in development, staging, and production.
# "Works on my machine" is eliminated with Docker.
#
# Multi-stage build:
#   Stage 1 (builder): Install dependencies
#   Stage 2 (runtime): Minimal production image
# ============================================================

# ── Stage 1: Builder ────────────────────────────────────────
FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --user -r requirements.txt

# ── Stage 2: Runtime ────────────────────────────────────────
FROM python:3.10-slim as runtime

# Security: Run as non-root user
RUN groupadd -r mlops && useradd -r -g mlops mlops

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY src/ ./src/
COPY api/ ./api/
COPY configs/ ./configs/
COPY monitoring/ ./monitoring/
COPY pipelines/ ./pipelines/

# Create directories for artifacts
RUN mkdir -p models reports mlruns data/raw data/processed && \
    chown -R mlops:mlops /app

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH=/root/.local/bin:$PATH
ENV MLFLOW_TRACKING_URI=mlruns

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Switch to non-root user
USER mlops

# Start the API server
CMD ["uvicorn", "api.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info"]
