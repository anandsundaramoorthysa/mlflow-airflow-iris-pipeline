"""Model definitions and training helpers for the Iris pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.config import MODELS_DIR, RANDOM_STATE

# Parameters logged to MLflow and passed to each estimator.
MODEL_PARAMS = {
    "logistic_regression": {
        "model_type": "LogisticRegression",
        "C": 1.0,
        "max_iter": 200,
        "random_state": RANDOM_STATE,
    },
    "random_forest": {
        "model_type": "RandomForestClassifier",
        "n_estimators": 100,
        "max_depth": None,
        "random_state": RANDOM_STATE,
    },
}

_MODEL_CLASSES = {
    "LogisticRegression": LogisticRegression,
    "RandomForestClassifier": RandomForestClassifier,
}


def build_model(model_name: str):
    """Build an uncompiled estimator from the parameter registry."""
    if model_name not in MODEL_PARAMS:
        raise ValueError(f"Unknown model: {model_name}")
    params = MODEL_PARAMS[model_name]
    estimator_cls = _MODEL_CLASSES[params["model_type"]]
    estimator_kwargs = {key: value for key, value in params.items() if key != "model_type"}
    return estimator_cls(**estimator_kwargs)


def train_model(model, X_train: pd.DataFrame, y_train: pd.Series):
    """Fit the estimator on the training data."""
    return model.fit(X_train, y_train)


def save_model(model, model_name: str, models_dir=MODELS_DIR) -> Path:
    """Persist a fitted model to disk as joblib."""
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"{model_name}.joblib"
    joblib.dump(model, path)
    return path


def load_model(model_name: str, models_dir=MODELS_DIR):
    """Load a previously fitted model from disk."""
    path = models_dir / f"{model_name}.joblib"
    return joblib.load(path)


def save_model_params(model_name: str, models_dir=MODELS_DIR) -> Path:
    """Persist the logged-parameter dict for a model."""
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"{model_name}_params.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(MODEL_PARAMS[model_name], handle, indent=2)
    return path


def load_model_params(model_name: str, models_dir=MODELS_DIR) -> dict:
    """Load the logged-parameter dict previously saved for a model."""
    path = models_dir / f"{model_name}_params.json"
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)