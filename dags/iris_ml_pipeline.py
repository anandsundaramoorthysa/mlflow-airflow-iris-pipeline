"""
Iris ML pipeline: Airflow orchestrates the workflow, MLflow tracks it.

Task flow
    load_data -> preprocess_data -> train_models -> evaluate_models

Responsibilities
    Airflow  : scheduling, task ordering, retries, logs
    MLflow   : experiment runs, params, metrics, models, artifacts
    Sklearn  : the actual machine learning

Run manually from the Airflow UI, or watch it fire on a @daily schedule.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pendulum
from airflow.sdk import DAG, task

# Make the reusable src/ package importable from the DAGs folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import data as data_ops
from src import evaluate as eval_ops
from src import train as train_ops
from src.config import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    RAW_DATA_PATH,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
)

doc_md = __doc__ or ""


@task
def load_data() -> str:
    """Load the Iris dataset and persist the raw CSV."""
    pandas_df = data_ops.load_iris()
    data_ops.save_dataset(pandas_df, RAW_DATA_PATH)
    print(f"Loaded {len(pandas_df)} samples -> {RAW_DATA_PATH}")
    return str(RAW_DATA_PATH)


@task
def preprocess_data(raw_path: str) -> str:
    """Split data into stratified train/test sets and persist both.

    Returns a single JSON string (Airflow 3 TaskFlow does not support
    unpacking tuple outputs) pointing at the two generated files.
    """
    raw_df = data_ops.load_dataset(raw_path)
    train_df, test_df = data_ops.split_dataset(raw_df)
    data_ops.save_dataset(train_df, TRAIN_DATA_PATH)
    data_ops.save_dataset(test_df, TEST_DATA_PATH)
    split_paths = json.dumps(
        {"train_path": str(TRAIN_DATA_PATH), "test_path": str(TEST_DATA_PATH)}
    )
    print(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows -> {split_paths}")
    return split_paths


@task
def train_models(split_paths: str) -> list[str]:
    """Fit each model in the registry and persist it locally."""
    split = json.loads(split_paths)
    train_df = data_ops.load_dataset(split["train_path"])
    X_train, y_train = data_ops.separate_features(train_df)

    model_names = list(train_ops.MODEL_PARAMS.keys())
    for model_name in model_names:
        model = train_ops.build_model(model_name)
        train_ops.train_model(model, X_train, y_train)
        train_ops.save_model(model, model_name)
        train_ops.save_model_params(model_name)
        print(f"Trained and saved model: {model_name}")

    return model_names


@task
def evaluate_models(model_names: list[str], split_paths: str) -> dict[str, str]:
    """Score every trained model and log a full MLflow run for each."""
    split = json.loads(split_paths)
    test_df = data_ops.load_dataset(split["test_path"])
    X_test, y_test = data_ops.separate_features(test_df)

    run_ids: dict[str, str] = {}
    for model_name in model_names:
        model = train_ops.load_model(model_name)
        params = train_ops.load_model_params(model_name)

        metrics, _ = eval_ops.evaluate_model(model, X_test, y_test)
        run_id = eval_ops.log_run(
            model_name=model_name,
            params=params,
            model=model,
            metrics=metrics,
            X_test=X_test,
            y_test=y_test,
            tracking_uri=MLFLOW_TRACKING_URI,
            experiment_name=MLFLOW_EXPERIMENT_NAME,
        )
        run_ids[model_name] = run_id
        print(f"{model_name}: accuracy={metrics['accuracy']:.4f}, run_id={run_id}")

    return run_ids


with DAG(
    dag_id="iris_ml_pipeline",
    description="Train and evaluate Iris classifiers, tracked end-to-end in MLflow.",
    doc_md=doc_md,
    schedule="@daily",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    tags=["iris", "mlflow", "classification"],
    default_args={
        "owner": "mlops",
        "retries": 1,
        "retry_delay": pendulum.duration(minutes=1),
    },
) as dag:
    raw_path = load_data()
    split_paths = preprocess_data(raw_path)
    trained_models = train_models(split_paths)
    evaluate_models(trained_models, split_paths)