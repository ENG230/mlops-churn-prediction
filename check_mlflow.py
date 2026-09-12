"""Check MLflow experiments and registered models."""
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = mlflow.tracking.MlflowClient()

print("=" * 60)
print("MLflow Tracking URI: sqlite:///mlflow.db")
print("=" * 60)

# List experiments
experiments = client.search_experiments()
print(f"\n=== EXPERIMENTS ({len(experiments)} found) ===")
for exp in experiments:
    print(f"\n  Experiment: '{exp.name}' (ID: {exp.experiment_id})")
    runs = client.search_runs(experiment_ids=[exp.experiment_id])
    print(f"  Total runs: {len(runs)}")
    for run in runs:
        m = run.data.metrics
        acc = m.get("test_accuracy", 0)
        f1 = m.get("test_f1", 0)
        auc = m.get("test_roc_auc", 0)
        algo = run.data.params.get("algorithm", "unknown")
        qg = run.data.params.get("quality_gate_passed", "unknown")
        print(f"\n    Run Name:  {run.info.run_name}")
        print(f"    Run ID:    {run.info.run_id}")
        print(f"    Algorithm: {algo}")
        print(f"    Accuracy:  {acc:.4f}")
        print(f"    F1 Score:  {f1:.4f}")
        print(f"    ROC-AUC:   {auc:.4f}")
        print(f"    Quality Gate: {qg}")

# List registered models
print("\n=== REGISTERED MODELS ===")
models = client.search_registered_models()
if not models:
    print("  No registered models found.")
for m in models:
    print(f"\n  Model: '{m.name}'")
    versions = client.search_model_versions(f"name='{m.name}'")
    for v in versions:
        print(f"    Version {v.version} | Status: {v.status} | Run: {v.run_id[:8]}...")

print("\n" + "=" * 60)
print("To view in browser: http://localhost:5000")
print("(MLflow UI must be running - see the separate PowerShell window)")
print("=" * 60)
