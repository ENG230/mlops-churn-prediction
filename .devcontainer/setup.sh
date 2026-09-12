#!/bin/bash
# ============================================================
# GitHub Codespaces Setup Script
# Runs automatically after the container is created
# ============================================================

set -e  # Exit on any error

echo "=============================================="
echo "  MLOps Churn Prediction - Codespaces Setup"
echo "=============================================="

# Install Python dependencies
echo ""
echo "📦 Installing Python dependencies..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
pip install -e . --quiet
echo "✅ Dependencies installed"

# Create necessary directories
echo ""
echo "📁 Creating project directories..."
mkdir -p data/raw data/processed models reports mlartifacts
echo "✅ Directories created"

# Run data ingestion to generate initial dataset
echo ""
echo "🔄 Generating synthetic dataset..."
python src/data_ingestion.py
echo "✅ Dataset generated"

echo ""
echo "=============================================="
echo "  Setup Complete! Ready to use."
echo "=============================================="
echo ""
echo "Quick Start Commands:"
echo "  1. Run full pipeline:    python pipelines/training_pipeline.py"
echo "  2. Start MLflow UI:      mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000"
echo "  3. Start API:            uvicorn api.app:app --reload --port 8000"
echo "  4. Build Docker image:   docker build -t churn-api ."
echo "  5. Run Docker container: docker run -p 8000:8000 -v \$(pwd)/models:/app/models churn-api"
echo "  6. Run tests:            pytest tests/ -v"
echo ""
