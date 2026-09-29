"""Central configuration for the project.

Resolves project-relative paths and MLflow settings. Environment
variables (optionally loaded from ``.env``) override the defaults so no
machine-specific paths are hardcoded anywhere in the codebase.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root = parent of src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load optional .env file (see .env.example). Does nothing if absent.
load_dotenv(PROJECT_ROOT / ".env")

# --- Directories -----------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

RAW_DATA_PATH = DATA_DIR / "iris.csv"
TRAIN_DATA_PATH = DATA_DIR / "train.csv"
TEST_DATA_PATH = DATA_DIR / "test.csv"

# --- Reproducibility -------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2

# --- MLflow ----------------------------------------------------------------
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "iris-classification")