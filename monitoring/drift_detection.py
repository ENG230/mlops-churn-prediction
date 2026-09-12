"""
Model & Data Drift Detection
==============================
Monitors production data for drift compared to training reference data.

MLOps Concepts Demonstrated:
- Data drift detection (feature distribution shift)
- Target/prediction drift detection
- Model performance degradation monitoring
- Automated retraining trigger logic
- HTML report generation with Evidently AI

Why Drift Monitoring Matters:
  Models degrade silently in production when:
  - Input data distribution changes (data drift)
  - The relationship between features and target changes (concept drift)
  - Business rules or customer behavior changes
  
  Without monitoring, you won't know until business metrics drop!
"""

import sys
import os
import logging
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional

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


def generate_drifted_data(reference_df: pd.DataFrame, drift_factor: float = 0.3) -> pd.DataFrame:
    """
    Simulate production data with injected drift.
    
    MLOps Concept: In real scenarios, this would be actual production data.
    We simulate drift to demonstrate monitoring capabilities.
    
    Drift injected:
    - tenure: shifted lower (more new customers)
    - MonthlyCharges: shifted higher (price increase)
    - Contract: more month-to-month (business change)
    """
    logger.info(f"Generating drifted production data (drift_factor={drift_factor})...")
    drifted = reference_df.copy()
    n = len(drifted)

    # Drift 1: Tenure shift (more new customers)
    if "tenure" in drifted.columns:
        tenure_drift = np.random.normal(-10 * drift_factor, 3, n)
        drifted["tenure"] = np.clip(drifted["tenure"] + tenure_drift, 0, 72).astype(int)

    # Drift 2: Monthly charges increase
    if "MonthlyCharges" in drifted.columns:
        charge_drift = np.random.normal(15 * drift_factor, 5, n)
        drifted["MonthlyCharges"] = np.clip(
            drifted["MonthlyCharges"] + charge_drift, 18.25, 118.75
        ).round(2)

    # Drift 3: More month-to-month contracts
    if "Contract" in drifted.columns:
        mask = np.random.random(n) < drift_factor * 0.5
        drifted.loc[mask, "Contract"] = "Month-to-month"

    # Drift 4: More fiber optic (infrastructure change)
    if "InternetService" in drifted.columns:
        mask = np.random.random(n) < drift_factor * 0.3
        drifted.loc[mask, "InternetService"] = "Fiber optic"

    logger.info(f"Drifted data generated: {len(drifted)} samples")
    return drifted


def compute_psi(reference: pd.Series, production: pd.Series, bins: int = 10) -> float:
    """
    Compute Population Stability Index (PSI) for numerical features.
    
    PSI measures how much a distribution has shifted:
    - PSI < 0.1:  No significant change
    - PSI 0.1-0.2: Moderate change, monitor closely
    - PSI > 0.2:  Significant change, investigate/retrain
    
    This is a lightweight alternative to Evidently for numerical drift.
    """
    # Create bins from reference data
    min_val = min(reference.min(), production.min())
    max_val = max(reference.max(), production.max())
    bin_edges = np.linspace(min_val, max_val, bins + 1)

    # Calculate proportions
    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    prod_counts, _ = np.histogram(production, bins=bin_edges)

    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    ref_pct = (ref_counts + epsilon) / (len(reference) + epsilon * bins)
    prod_pct = (prod_counts + epsilon) / (len(production) + epsilon * bins)

    # PSI formula
    psi = np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct))
    return round(float(psi), 4)


def compute_categorical_drift(reference: pd.Series, production: pd.Series) -> dict:
    """
    Compute distribution shift for categorical features.
    Returns chi-square-like statistic and distribution comparison.
    """
    ref_dist = reference.value_counts(normalize=True).to_dict()
    prod_dist = production.value_counts(normalize=True).to_dict()

    all_categories = set(ref_dist.keys()) | set(prod_dist.keys())
    max_shift = 0
    shifts = {}

    for cat in all_categories:
        ref_pct = ref_dist.get(cat, 0)
        prod_pct = prod_dist.get(cat, 0)
        shift = abs(prod_pct - ref_pct)
        shifts[cat] = {
            "reference_pct": round(ref_pct, 4),
            "production_pct": round(prod_pct, 4),
            "shift": round(shift, 4)
        }
        max_shift = max(max_shift, shift)

    return {
        "max_shift": round(max_shift, 4),
        "categories": shifts
    }


def run_drift_detection(
    reference_path: Optional[str] = None,
    production_data: Optional[pd.DataFrame] = None,
    config_path: str = "configs/params.yaml",
    simulate_drift: bool = True,
    drift_factor: float = 0.3,
    output_dir: str = "reports"
) -> dict:
    """
    Main drift detection function.
    
    Compares reference (training) data against production data.
    Generates a comprehensive drift report.
    
    Args:
        reference_path: Path to reference (training) data CSV
        production_data: Production DataFrame (if already loaded)
        config_path: Config file path
        simulate_drift: Whether to simulate drift for demo
        drift_factor: How much drift to inject (0-1)
        output_dir: Directory to save reports
    
    Returns:
        drift_report: Dictionary with drift metrics and alerts
    """
    config = load_config(config_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load reference data
    if reference_path is None:
        reference_path = config["monitoring"]["reference_data_path"]

    if not Path(reference_path).exists():
        logger.warning(f"Reference data not found at {reference_path}")
        logger.info("Running data ingestion to generate reference data...")
        from src.data_ingestion import ingest_data
        from src.feature_engineering import preprocess_raw_data, split_and_save_data
        raw_df = ingest_data(config_path)
        processed_df = preprocess_raw_data(raw_df, config)
        train_df, _ = split_and_save_data(processed_df, config)
        reference_df = train_df
    else:
        reference_df = pd.read_csv(reference_path)
        logger.info(f"Reference data loaded: {len(reference_df)} samples")

    # Get or simulate production data
    if production_data is None:
        if simulate_drift:
            production_df = generate_drifted_data(reference_df, drift_factor)
        else:
            # Use test data as production proxy
            test_path = config["data"]["processed_test_path"]
            if Path(test_path).exists():
                production_df = pd.read_csv(test_path)
            else:
                production_df = reference_df.sample(frac=0.3, random_state=99)
    else:
        production_df = production_data

    logger.info(f"Production data: {len(production_df)} samples")

    # ── Numerical Feature Drift (PSI) ────────────────────────
    numerical_features = config["features"]["numerical_features"]
    numerical_drift = {}

    logger.info("\n📊 NUMERICAL FEATURE DRIFT (PSI):")
    logger.info("-" * 50)
    for feature in numerical_features:
        if feature in reference_df.columns and feature in production_df.columns:
            ref_series = pd.to_numeric(reference_df[feature], errors="coerce").dropna()
            prod_series = pd.to_numeric(production_df[feature], errors="coerce").dropna()
            psi = compute_psi(ref_series, prod_series)

            if psi < 0.1:
                status = "✅ STABLE"
            elif psi < 0.2:
                status = "⚠️  MODERATE"
            else:
                status = "🚨 SIGNIFICANT"

            numerical_drift[feature] = {
                "psi": psi,
                "status": status.strip(),
                "ref_mean": round(float(ref_series.mean()), 4),
                "prod_mean": round(float(prod_series.mean()), 4),
                "ref_std": round(float(ref_series.std()), 4),
                "prod_std": round(float(prod_series.std()), 4),
            }
            logger.info(f"  {feature}: PSI={psi:.4f} | {status}")
            logger.info(f"    Ref mean: {ref_series.mean():.2f} → Prod mean: {prod_series.mean():.2f}")

    # ── Categorical Feature Drift ────────────────────────────
    categorical_features = config["features"]["categorical_features"]
    categorical_drift = {}

    logger.info("\n📊 CATEGORICAL FEATURE DRIFT:")
    logger.info("-" * 50)
    for feature in categorical_features[:5]:  # Show top 5 for brevity
        if feature in reference_df.columns and feature in production_df.columns:
            drift_info = compute_categorical_drift(
                reference_df[feature], production_df[feature]
            )
            max_shift = drift_info["max_shift"]

            if max_shift < 0.05:
                status = "✅ STABLE"
            elif max_shift < 0.15:
                status = "⚠️  MODERATE"
            else:
                status = "🚨 SIGNIFICANT"

            categorical_drift[feature] = {
                "max_shift": max_shift,
                "status": status.strip(),
                "details": drift_info["categories"]
            }
            logger.info(f"  {feature}: Max shift={max_shift:.4f} | {status}")

    # ── Target/Prediction Drift ──────────────────────────────
    target = config["data"]["target_column"]
    target_drift = {}

    if target in reference_df.columns and target in production_df.columns:
        # Handle both string and numeric target
        ref_target = reference_df[target]
        prod_target = production_df[target]

        if ref_target.dtype == object:
            ref_churn_rate = (ref_target == "Yes").mean()
            prod_churn_rate = (prod_target == "Yes").mean()
        else:
            ref_churn_rate = ref_target.mean()
            prod_churn_rate = prod_target.mean()

        churn_shift = abs(prod_churn_rate - ref_churn_rate)

        if churn_shift < 0.05:
            target_status = "✅ STABLE"
        elif churn_shift < 0.10:
            target_status = "⚠️  MODERATE"
        else:
            target_status = "🚨 SIGNIFICANT"

        target_drift = {
            "ref_churn_rate": round(float(ref_churn_rate), 4),
            "prod_churn_rate": round(float(prod_churn_rate), 4),
            "absolute_shift": round(float(churn_shift), 4),
            "status": target_status.strip()
        }

        logger.info(f"\n📊 TARGET DRIFT:")
        logger.info(f"  Churn rate: {ref_churn_rate:.2%} (ref) → {prod_churn_rate:.2%} (prod)")
        logger.info(f"  Shift: {churn_shift:.4f} | {target_status}")

    # ── Overall Drift Assessment ─────────────────────────────
    significant_numerical = sum(
        1 for v in numerical_drift.values() if v["psi"] >= 0.2
    )
    moderate_numerical = sum(
        1 for v in numerical_drift.values() if 0.1 <= v["psi"] < 0.2
    )
    significant_categorical = sum(
        1 for v in categorical_drift.values() if v["max_shift"] >= 0.15
    )

    drift_detected = (significant_numerical > 0 or significant_categorical > 0 or
                      target_drift.get("absolute_shift", 0) >= 0.10)

    retraining_recommended = (
        significant_numerical >= 2 or
        significant_categorical >= 2 or
        target_drift.get("absolute_shift", 0) >= 0.10
    )

    # ── Build Report ─────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    drift_report = {
        "report_id": f"drift_{timestamp}",
        "timestamp": datetime.now().isoformat(),
        "reference_samples": len(reference_df),
        "production_samples": len(production_df),
        "drift_simulated": simulate_drift,
        "drift_factor": drift_factor if simulate_drift else None,
        "numerical_drift": numerical_drift,
        "categorical_drift": categorical_drift,
        "target_drift": target_drift,
        "summary": {
            "significant_numerical_features": significant_numerical,
            "moderate_numerical_features": moderate_numerical,
            "significant_categorical_features": significant_categorical,
            "drift_detected": drift_detected,
            "retraining_recommended": retraining_recommended,
        }
    }

    # ── Save JSON Report ─────────────────────────────────────
    report_path = f"{output_dir}/drift_report_{timestamp}.json"
    with open(report_path, "w") as f:
        json.dump(drift_report, f, indent=2)
    logger.info(f"\n📄 Drift report saved to: {report_path}")

    # ── Generate HTML Report ─────────────────────────────────
    html_path = generate_html_drift_report(drift_report, output_dir, timestamp)

    # ── Log Summary ──────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("DRIFT DETECTION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Significant numerical drift: {significant_numerical} features")
    logger.info(f"  Moderate numerical drift:    {moderate_numerical} features")
    logger.info(f"  Significant categorical drift: {significant_categorical} features")
    logger.info(f"  Target drift: {target_drift.get('absolute_shift', 0):.4f}")
    logger.info(f"  Drift detected: {'YES 🚨' if drift_detected else 'NO ✅'}")
    logger.info(f"  Retraining recommended: {'YES 🔄' if retraining_recommended else 'NO'}")
    logger.info("=" * 60)

    if retraining_recommended:
        logger.warning("⚠️  RETRAINING RECOMMENDED!")
        logger.warning("   Run: python pipelines/training_pipeline.py")

    return drift_report


def generate_html_drift_report(report: dict, output_dir: str, timestamp: str) -> str:
    """Generate a simple HTML drift report for visualization."""
    html_path = f"{output_dir}/drift_report_{timestamp}.html"

    numerical_rows = ""
    for feature, info in report.get("numerical_drift", {}).items():
        status_color = (
            "#28a745" if "STABLE" in info["status"] else
            "#ffc107" if "MODERATE" in info["status"] else "#dc3545"
        )
        numerical_rows += f"""
        <tr>
            <td><strong>{feature}</strong></td>
            <td>{info['psi']:.4f}</td>
            <td>{info['ref_mean']:.2f}</td>
            <td>{info['prod_mean']:.2f}</td>
            <td style="color:{status_color}; font-weight:bold">{info['status']}</td>
        </tr>"""

    categorical_rows = ""
    for feature, info in report.get("categorical_drift", {}).items():
        status_color = (
            "#28a745" if "STABLE" in info["status"] else
            "#ffc107" if "MODERATE" in info["status"] else "#dc3545"
        )
        categorical_rows += f"""
        <tr>
            <td><strong>{feature}</strong></td>
            <td>{info['max_shift']:.4f}</td>
            <td style="color:{status_color}; font-weight:bold">{info['status']}</td>
        </tr>"""

    summary = report.get("summary", {})
    target = report.get("target_drift", {})
    drift_color = "#dc3545" if summary.get("drift_detected") else "#28a745"
    retrain_color = "#dc3545" if summary.get("retraining_recommended") else "#28a745"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MLOps Drift Detection Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1100px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin: 20px 0; }}
        .metric-card {{ background: #f8f9fa; border-radius: 8px; padding: 15px; text-align: center; border-left: 4px solid #3498db; }}
        .metric-card .value {{ font-size: 2em; font-weight: bold; color: #2c3e50; }}
        .metric-card .label {{ color: #7f8c8d; font-size: 0.9em; margin-top: 5px; }}
        .alert {{ padding: 15px; border-radius: 8px; margin: 15px 0; font-weight: bold; }}
        .alert-danger {{ background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }}
        .alert-success {{ background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }}
        .alert-warning {{ background: #fff3cd; color: #856404; border: 1px solid #ffeeba; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th {{ background: #3498db; color: white; padding: 10px; text-align: left; }}
        td {{ padding: 8px 10px; border-bottom: 1px solid #dee2e6; }}
        tr:hover {{ background: #f8f9fa; }}
        .badge {{ padding: 3px 8px; border-radius: 12px; font-size: 0.85em; }}
        .footer {{ margin-top: 30px; color: #7f8c8d; font-size: 0.85em; text-align: center; }}
    </style>
</head>
<body>
<div class="container">
    <h1>🔍 MLOps Drift Detection Report</h1>
    <p><strong>Report ID:</strong> {report['report_id']} | 
       <strong>Generated:</strong> {report['timestamp']} |
       <strong>Reference:</strong> {report['reference_samples']:,} samples |
       <strong>Production:</strong> {report['production_samples']:,} samples
    </p>

    {"<div class='alert alert-warning'>⚠️ Note: This report uses SIMULATED drift data for demonstration purposes.</div>" if report.get('drift_simulated') else ""}

    <div class="summary-grid">
        <div class="metric-card">
            <div class="value" style="color:{drift_color}">{'YES' if summary.get('drift_detected') else 'NO'}</div>
            <div class="label">Drift Detected</div>
        </div>
        <div class="metric-card">
            <div class="value" style="color:{retrain_color}">{'YES' if summary.get('retraining_recommended') else 'NO'}</div>
            <div class="label">Retraining Recommended</div>
        </div>
        <div class="metric-card">
            <div class="value">{summary.get('significant_numerical_features', 0)}</div>
            <div class="label">Features with Significant Drift</div>
        </div>
    </div>

    {"<div class='alert alert-danger'>🚨 SIGNIFICANT DRIFT DETECTED! Retraining is recommended.</div>" if summary.get('retraining_recommended') else "<div class='alert alert-success'>✅ No significant drift detected. Model is stable.</div>"}

    <h2>📊 Numerical Feature Drift (PSI)</h2>
    <p>PSI &lt; 0.1: Stable | PSI 0.1-0.2: Moderate | PSI &gt; 0.2: Significant</p>
    <table>
        <tr><th>Feature</th><th>PSI Score</th><th>Ref Mean</th><th>Prod Mean</th><th>Status</th></tr>
        {numerical_rows}
    </table>

    <h2>📊 Categorical Feature Drift</h2>
    <table>
        <tr><th>Feature</th><th>Max Distribution Shift</th><th>Status</th></tr>
        {categorical_rows}
    </table>

    <h2>🎯 Target Drift</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Reference Churn Rate</td><td>{target.get('ref_churn_rate', 0):.2%}</td></tr>
        <tr><td>Production Churn Rate</td><td>{target.get('prod_churn_rate', 0):.2%}</td></tr>
        <tr><td>Absolute Shift</td><td>{target.get('absolute_shift', 0):.4f}</td></tr>
        <tr><td>Status</td><td style="font-weight:bold">{target.get('status', 'N/A')}</td></tr>
    </table>

    <h2>🔄 Recommended Actions</h2>
    <ul>
        {"<li>🚨 <strong>Retrain model</strong>: Run <code>python pipelines/training_pipeline.py</code></li>" if summary.get('retraining_recommended') else ""}
        {"<li>⚠️ Monitor closely: Moderate drift detected in some features</li>" if summary.get('moderate_numerical_features', 0) > 0 else ""}
        <li>📊 Review feature distributions in detail</li>
        <li>🔍 Investigate business changes that may explain drift</li>
        <li>📅 Schedule next drift check in 7 days</li>
    </ul>

    <div class="footer">
        <p>Generated by MLOps Churn Prediction Monitoring System | 
           <a href="/docs">API Docs</a> | 
           <a href="http://localhost:5000">MLflow UI</a>
        </p>
    </div>
</div>
</body>
</html>"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    logger.info(f"HTML drift report saved to: {html_path}")
    return html_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Drift Detection")
    parser.add_argument("--config", default="configs/params.yaml")
    parser.add_argument("--drift-factor", type=float, default=0.3,
                        help="Drift factor for simulation (0-1)")
    parser.add_argument("--no-simulate", action="store_true",
                        help="Use actual test data instead of simulated drift")
    args = parser.parse_args()

    report = run_drift_detection(
        config_path=args.config,
        simulate_drift=not args.no_simulate,
        drift_factor=args.drift_factor
    )

    print(f"\n{'='*50}")
    print(f"Drift Detection Complete!")
    print(f"Drift Detected: {report['summary']['drift_detected']}")
    print(f"Retraining Recommended: {report['summary']['retraining_recommended']}")
    print(f"\nReports saved to: reports/")
