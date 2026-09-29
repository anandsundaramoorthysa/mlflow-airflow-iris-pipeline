"""Tests for the Iris ML pipeline: data, training, evaluation, and the DAG."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Configure Airflow before anything imports it so the DAG import test uses a
# deterministic, project-local AIRFLOW_HOME.
os.environ.setdefault("AIRFLOW_HOME", str(PROJECT_ROOT / ".airflow"))
os.environ.setdefault("AIRFLOW__CORE__LOAD_EXAMPLES", "False")
os.environ.setdefault("AIRFLOW__CORE__EXECUTOR", "LocalExecutor")

sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest

from src import data as data_ops
from src import train as train_ops

pytest.importorskip("mlflow")
from src import evaluate as eval_ops


# --- Dataset ---------------------------------------------------------------

def test_load_iris_shape_and_validity():
    df = data_ops.load_iris()
    assert df.shape == (150, 5)
    assert list(df.columns) == data_ops.FEATURE_COLUMNS + [data_ops.TARGET_COLUMN]
    assert not df.isna().any().any()
    assert set(df[data_ops.TARGET_COLUMN].unique()) == {0, 1, 2}


def test_split_dataset_shapes_and_stratification():
    df = data_ops.load_iris()
    train, test = data_ops.split_dataset(df)
    assert len(train) == 120
    assert len(test) == 30
    assert df.index.isin(test.index).any()
    assert not set(train.index) & set(test.index)
    train_ratio = train[data_ops.TARGET_COLUMN].mean()
    test_ratio = test[data_ops.TARGET_COLUMN].mean()
    assert abs(train_ratio - test_ratio) < 0.05


def test_split_is_reproducible():
    df = data_ops.load_iris()
    train_a, test_a = data_ops.split_dataset(df)
    train_b, test_b = data_ops.split_dataset(df)
    assert train_a[data_ops.TARGET_COLUMN].tolist() == train_b[data_ops.TARGET_COLUMN].tolist()
    assert test_a.index.tolist() == test_b.index.tolist()


# --- Training --------------------------------------------------------------

@pytest.mark.parametrize("model_name", list(train_ops.MODEL_PARAMS.keys()))
def test_models_train_and_predict(model_name):
    df = data_ops.load_iris()
    train, _ = data_ops.split_dataset(df)
    X_train, y_train = data_ops.separate_features(train)

    model = train_ops.build_model(model_name)
    train_ops.train_model(model, X_train, y_train)

    predictions = model.predict(X_train)
    assert predictions.shape == (len(X_train),)
    assert set(np.unique(predictions)) <= {0, 1, 2}


@pytest.mark.parametrize("model_name", list(train_ops.MODEL_PARAMS.keys()))
def test_registered_params_fit(model_name):
    params = train_ops.MODEL_PARAMS[model_name]
    assert params["random_state"] == 42
    assert params["model_type"] in {"LogisticRegression", "RandomForestClassifier"}


# --- Evaluation ------------------------------------------------------------

@pytest.mark.parametrize("model_name", list(train_ops.MODEL_PARAMS.keys()))
def test_evaluate_metrics_are_bounded(model_name):
    df = data_ops.load_iris()
    train, test = data_ops.split_dataset(df)
    X_train, y_train = data_ops.separate_features(train)
    X_test, y_test = data_ops.separate_features(test)

    model = train_ops.build_model(model_name)
    train_ops.train_model(model, X_train, y_train)

    metrics, predictions = eval_ops.evaluate_model(model, X_test, y_test)
    expected = {"accuracy", "precision", "recall", "f1_score"}
    assert set(metrics) == expected
    assert predictions.shape == (len(X_test),)
    for value in metrics.values():
        assert 0.0 <= value <= 1.0


def test_classification_report_contains_target_names():
    df = data_ops.load_iris()
    train, test = data_ops.split_dataset(df)
    X_train, y_train = data_ops.separate_features(train)
    X_test, y_test = data_ops.separate_features(test)
    model = train_ops.build_model("logistic_regression")
    train_ops.train_model(model, X_train, y_train)

    predictions = model.predict(X_test)
    report = eval_ops.build_classification_report(y_test, predictions)
    assert "setosa" in report and "versicolor" in report and "virginica" in report


# --- Airflow DAG -----------------------------------------------------------

def test_dag_imports_and_structure():
    pytest.importorskip("airflow")
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT / "dags"))
    module = importlib.import_module("iris_ml_pipeline")
    dag = getattr(module, "dag")
    assert dag.dag_id == "iris_ml_pipeline"
    assert {task_id for task_id in dag.task_dict} == {
        "load_data",
        "preprocess_data",
        "train_models",
        "evaluate_models",
    }
    dependencies = {task_id: [t.task_id for t in dag.get_task(task_id).upstream_list] for task_id in dag.task_dict}
    assert dependencies["preprocess_data"] == ["load_data"]
    assert "train_models" in dependencies["evaluate_models"]
    assert "preprocess_data" in dependencies["train_models"]