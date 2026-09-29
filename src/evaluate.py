"""Model evaluation and MLflow experiment-logging helpers."""

from __future__ import annotations

import json

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

CLASS_NAMES = ["setosa", "versicolor", "virginica"]


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Score a fitted model on the held-out test set.

    Macro-averaging is used for precision/recall/f1 because the task is
    multiclass and we want each class to count equally.
    Returns (metrics dict, raw predictions).
    """
    predictions = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, average="macro"),
        "recall": recall_score(y_test, predictions, average="macro"),
        "f1_score": f1_score(y_test, predictions, average="macro"),
    }
    return metrics, predictions


def build_classification_report(y_test: pd.Series, predictions) -> str:
    """Human-readable classification report as a string artifact."""
    return classification_report(
        y_test, predictions, target_names=CLASS_NAMES, digits=4, zero_division=0
    )


def confusion_matrix_json(y_test: pd.Series, predictions) -> dict:
    """Confusion matrix as a JSON-serialisable structure."""
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1, 2])
    return {
        "target_names": CLASS_NAMES,
        "matrix": [[int(value) for value in row] for row in matrix.tolist()],
    }


def log_run(
    model_name: str,
    params: dict,
    model,
    metrics: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    tracking_uri: str = "http://localhost:5000",
    experiment_name: str = "iris-classification",
) -> str:
    """
    Create one MLflow run for a model and log params, metrics, the model,
    and evaluation artifacts. Any exception propagates so the DAG task
    fails (and MLflow marks the run as FAILED).
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    predictions = model.predict(X_test)

    with mlflow.start_run(run_name=model_name) as run:
        mlflow.set_tag("pipeline", "airflow")
        mlflow.set_tag("model_type", params["model_type"])

        # Params must be json-serialisable strings for MLflow.
        safe_params = {key: json.dumps(value) if value is None else str(value) for key, value in params.items()}
        mlflow.log_params(safe_params)

        signature = infer_signature(X_test, predictions[:5])
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            # skops (MLflow's default sklearn serializer) distrusts raw tree
            # storage; we own the model, so mark it as trusted.
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        mlflow.log_metrics(metrics)

        report = build_classification_report(y_test, predictions)
        mlflow.log_text(report, "classification_report.txt")
        mlflow.log_dict(confusion_matrix_json(y_test, predictions), "confusion_matrix.json")

    return run.info.run_id