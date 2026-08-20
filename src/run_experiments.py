"""
Runs several meaningfully different model configurations (different algorithms
and different hyperparameters, not just random seed changes) and logs each
as a separate MLflow run under the 'heart-disease-prediction' experiment.

Usage:
    python src/run_experiments.py
"""
import os
import copy
import yaml

from train import train_model, log_run

BASE_DATA_CONFIG = {
    "raw_data_path": "data/heart_disease.csv",
    "test_size": 0.2,
    "random_state": 42,
}
TRAINING_CONFIG = {"target_column": "target"}
METRICS_CONFIG = {"primary_metric": "f1_score", "minimum_f1": 0.60}

EXPERIMENT_CONFIGS = [
    {
        "run_name": "rf_baseline",
        "model": {
            "type": "RandomForestClassifier",
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "class_weight": "balanced",
            "n_jobs": -1,
        },
    },
    {
        "run_name": "rf_deep_wide",
        "model": {
            "type": "RandomForestClassifier",
            "n_estimators": 300,
            "max_depth": 12,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "class_weight": "balanced",
            "n_jobs": -1,
        },
    },
    {
        "run_name": "rf_shallow_regularized",
        "model": {
            "type": "RandomForestClassifier",
            "n_estimators": 200,
            "max_depth": 4,
            "min_samples_split": 8,
            "min_samples_leaf": 4,
            "class_weight": "balanced",
            "n_jobs": -1,
        },
    },
    {
        "run_name": "gradient_boosting",
        "model": {
            "type": "GradientBoostingClassifier",
            "n_estimators": 150,
            "max_depth": 3,
            "learning_rate": 0.05,
            "min_samples_leaf": 2,
        },
    },
    {
        "run_name": "logistic_regression",
        "model": {
            "type": "LogisticRegression",
            "C": 1.0,
            "max_iter": 1000,
            "class_weight": "balanced",
        },
    },
    {
        "run_name": "svc_rbf",
        "model": {
            "type": "SVC",
            "C": 2.0,
            "kernel": "rbf",
            "gamma": "scale",
            "class_weight": "balanced",
        },
    },
    {
        "run_name": "knn",
        "model": {
            "type": "KNeighborsClassifier",
            "n_neighbors": 11,
            "weights": "distance",
        },
    },
]


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)  # so relative data path in config resolves correctly

    results = []
    for exp in EXPERIMENT_CONFIGS:
        config = {
            "data": copy.deepcopy(BASE_DATA_CONFIG),
            "model": exp["model"],
            "training": TRAINING_CONFIG,
            "metrics": METRICS_CONFIG,
        }

        model, metrics, feature_columns = train_model(config)
        run_id = log_run(model, metrics, config, feature_columns, run_name=exp["run_name"])

        results.append((exp["run_name"], run_id, metrics))
        print(f"[{exp['run_name']}] run_id={run_id} "
              f"accuracy={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} "
              f"roc_auc={metrics.get('roc_auc')}")

    print(f"\nLogged {len(results)} runs to MLflow experiment 'heart-disease-prediction'.")
    print("Run `mlflow ui` and open http://localhost:5000 to inspect them, "
          "or run `python src/compare_experiments.py` to pick the best.")


if __name__ == "__main__":
    main()
