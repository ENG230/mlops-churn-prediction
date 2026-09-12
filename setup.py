"""
Package setup for MLOps Churn Prediction project.
Allows the project to be installed as a package: pip install -e .
"""
from setuptools import setup, find_packages

setup(
    name="mlops-churn-prediction",
    version="1.0.0",
    description="End-to-End MLOps Demo: Customer Churn Prediction",
    author="MLOps Engineer",
    packages=find_packages(exclude=["tests*", "notebooks*"]),
    python_requires=">=3.10",
    install_requires=[
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "mlflow>=2.9.0",
        "fastapi>=0.109.0",
        "uvicorn>=0.27.0",
        "pydantic>=2.5.0",
        "pyyaml>=6.0.0",
        "joblib>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "httpx>=0.26.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
        ]
    },
)
