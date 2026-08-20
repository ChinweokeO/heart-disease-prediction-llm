import sys
import os
import yaml
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

src_root = os.path.dirname(os.path.abspath(__file__))
if src_root not in sys.path:
    sys.path.insert(0, src_root)

from preprocessing import load_data, preprocess_data, split_data
from evaluation import evaluate_model

# Local SQLite-backed MLflow tracking store (relative to repo root).
# MLflow 3.x deprecated the plain filesystem store ("mlruns") in favor of a
# database backend, so we point at a local sqlite file instead.
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("heart-disease-prediction")


def load_config(config_path):
    """Load the YAML configuration file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def build_model(model_config, random_state):
    """
    Build a model (or pipeline) from a model config dict.
    Supports several algorithm families so experiments can be
    meaningfully different, not just hyperparameter tweaks.
    """
    model_type = model_config.get("type", "RandomForestClassifier")

    if model_type == "RandomForestClassifier":
        return RandomForestClassifier(
            n_estimators=model_config.get("n_estimators", 200),
            max_depth=model_config.get("max_depth", None),
            min_samples_split=model_config.get("min_samples_split", 5),
            min_samples_leaf=model_config.get("min_samples_leaf", 2),
            class_weight=model_config.get("class_weight", "balanced"),
            n_jobs=model_config.get("n_jobs", -1),
            random_state=random_state,
        )

    elif model_type == "GradientBoostingClassifier":
        return GradientBoostingClassifier(
            n_estimators=model_config.get("n_estimators", 150),
            max_depth=model_config.get("max_depth", 3),
            learning_rate=model_config.get("learning_rate", 0.1),
            min_samples_leaf=model_config.get("min_samples_leaf", 2),
            random_state=random_state,
        )

    elif model_type == "LogisticRegression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=model_config.get("C", 1.0),
                max_iter=model_config.get("max_iter", 1000),
                class_weight=model_config.get("class_weight", "balanced"),
                random_state=random_state,
            )),
        ])

    elif model_type == "SVC":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                C=model_config.get("C", 1.0),
                kernel=model_config.get("kernel", "rbf"),
                gamma=model_config.get("gamma", "scale"),
                class_weight=model_config.get("class_weight", "balanced"),
                probability=True,
                random_state=random_state,
            )),
        ])

    elif model_type == "KNeighborsClassifier":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(
                n_neighbors=model_config.get("n_neighbors", 15),
                weights=model_config.get("weights", "distance"),
            )),
        ])

    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def train_model(config):
    """Train a model using configuration values. Returns model, metrics, and feature columns."""
    target_column = config["training"]["target_column"]
    df = load_data(config["data"]["raw_data_path"], target_column=target_column)

    numeric_columns = df.select_dtypes(include="number").columns.drop(target_column).tolist()
    categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()

    df = preprocess_data(df, numeric_columns, categorical_columns)

    X_train, X_test, y_train, y_test = split_data(
        df,
        target_column,
        config["data"]["test_size"],
        config["data"]["random_state"],
    )

    model = build_model(config["model"], config["data"]["random_state"])
    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)
    feature_columns = X_train.columns.tolist()

    return model, metrics, feature_columns


def log_run(model, metrics, config, feature_columns, run_name=None):
    """Log a single training run to MLflow. Returns the run_id."""
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(config["model"])
        mlflow.log_param("data_version", "heart_v1")
        mlflow.log_param("test_size", config["data"]["test_size"])
        mlflow.log_param("random_state", config["data"]["random_state"])
        mlflow.log_metrics({k: v for k, v in metrics.items() if v is not None})
        mlflow.sklearn.log_model(
            model, "model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )
        mlflow.log_dict({"feature_columns": feature_columns}, "feature_columns.json")
        return run.info.run_id


if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(repo_root, "configs", "config.yaml")
    config = load_config(config_path)

    model, metrics, feature_columns = train_model(config)
    run_id = log_run(model, metrics, config, feature_columns,
                      run_name=config["model"].get("type", "RandomForestClassifier"))

    print(f"Logged MLflow run: {run_id}")
    print("Model metrics:")
    print(f"  accuracy: {metrics['accuracy']:.4f}")
    print(f"  f1: {metrics['f1']:.4f}")
    if metrics.get("roc_auc") is not None:
        print(f"  roc_auc: {metrics['roc_auc']:.4f}")
    else:
        print("  roc_auc: not available")

    if metrics["f1"] < config["metrics"]["minimum_f1"]:
        raise ValueError("Model failed minimum F1 threshold")
