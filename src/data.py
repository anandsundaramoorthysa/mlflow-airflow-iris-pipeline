"""Reusable data loading and preprocessing helpers for the Iris dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.datasets import load_iris as load_iris_dataset
from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE, TEST_SIZE

FEATURE_COLUMNS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]
TARGET_COLUMN = "target"


def load_iris() -> pd.DataFrame:
    """Load the Iris dataset as a DataFrame with feature + target columns."""
    data = load_iris_dataset()
    df = pd.DataFrame(data["data"], columns=FEATURE_COLUMNS)
    df[TARGET_COLUMN] = data["target"]
    return df


def load_dataset(path) -> pd.DataFrame:
    """Read a previously saved dataset from a CSV file."""
    return pd.read_csv(path)


def save_dataset(df: pd.DataFrame, path) -> None:
    """Persist a DataFrame to disk as CSV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def separate_features(df: pd.DataFrame):
    """Split a DataFrame into features (X) and target (y)."""
    return df[FEATURE_COLUMNS], df[TARGET_COLUMN]


def split_dataset(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split into stratified train/test folds with a fixed random seed.

    The fixed ``random_state`` makes the split reproducible, and disabling
    shuffling concerns is handled by ``train_test_split`` internally.
    """
    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[TARGET_COLUMN],
        shuffle=True,
    )
    return train, test