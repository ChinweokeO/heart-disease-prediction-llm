"""
Queries all logged MLflow runs, ranks them by the primary metric (f1),
prints a comparison table, and exports the best model + its feature
column order to models/ so the LLM interface (app.py) can load it
without depending on MLflow at inference time.

Usage:
    python src/compare_experiments.py
"""
import os
import json
import mlflow
import joblib
import pandas as pd

PRIMARY_METRIC = "metrics.f1"
EXPERIMENT_NAME = "heart-disease-prediction"


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        order_by=[f"{PRIMARY_METRIC} DESC"],
    )

    if runs.empty:
        print("No runs found. Run `python src/run_experiments.py` first.")
        return

    display_cols = [
        "tags.mlflow.runName", "run_id",
        "metrics.accuracy", "metrics.f1", "metrics.roc_auc",
        "params.type",
    ]
    display_cols = [c for c in display_cols if c in runs.columns]

    print("\n=== Experiment comparison (sorted by F1, best first) ===")
    print(runs[display_cols].to_string(index=False))

    best = runs.iloc[0]
    best_run_id = best["run_id"]
    best_name = best.get("tags.mlflow.runName", best_run_id)
    print(f"\nBest run: {best_name} (run_id={best_run_id}) "
          f"f1={best['metrics.f1']:.4f} accuracy={best['metrics.accuracy']:.4f}")

    # Load and export the best model for the LLM interface to use directly.
    model_uri = f"runs:/{best_run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)

    feature_columns = mlflow.artifacts.load_dict(
        f"runs:/{best_run_id}/feature_columns.json"
    )["feature_columns"]

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/best_model.joblib")
    with open("models/feature_columns.json", "w") as f:
        json.dump(feature_columns, f, indent=2)
    with open("models/best_run_info.json", "w") as f:
        json.dump({
            "run_id": best_run_id,
            "run_name": best_name,
            "accuracy": best["metrics.accuracy"],
            "f1": best["metrics.f1"],
            "roc_auc": best.get("metrics.roc_auc"),
        }, f, indent=2)

    # Population-level defaults (median) for every feature. Used by the LLM
    # interface to fill in auxiliary fields a user is unlikely to know off
    # the top of their head (e.g. 'ca', 'thal'), always disclosed to the
    # user as an assumption rather than silently guessed.
    raw_df = pd.read_csv("data/heart_disease.csv")
    feature_defaults = {
        col: float(raw_df[col].median())
        for col in feature_columns if col in raw_df.columns
    }
    with open("models/feature_defaults.json", "w") as f:
        json.dump(feature_defaults, f, indent=2)

    print("\nExported best model -> models/best_model.joblib")
    print("Exported feature columns -> models/feature_columns.json")
    print("Exported feature defaults -> models/feature_defaults.json")


if __name__ == "__main__":
    main()
