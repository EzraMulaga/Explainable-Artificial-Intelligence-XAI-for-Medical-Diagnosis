"""
model.py
--------
Trains two RandomForestClassifier models:
  1. Diagnosis model  — binary classification (0 = Negative, 1 = Positive)
  2. Status model     — multi-class classification (0=Stable, 1=At Risk, 2=Critical)

Provides helpers for training, evaluation, and persistence.
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


# Default hyper-parameters
DEFAULT_RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 10,
    "min_samples_split": 4,
    "random_state": 42,
    "n_jobs": -1,
}

STATUS_LABELS = ["Stable", "At Risk", "Critical"]
DIAGNOSIS_LABELS = ["Negative", "Positive"]


def train_diagnosis_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict = None,
) -> RandomForestClassifier:
    """Train the binary diagnosis classifier."""
    params = params or DEFAULT_RF_PARAMS
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model


def train_status_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict = None,
) -> RandomForestClassifier:
    """Train the multi-class patient status classifier."""
    params = params or DEFAULT_RF_PARAMS
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target_names: list[str],
) -> dict:
    """
    Evaluate a trained model on the test split.

    Returns a dict with keys: accuracy, report, confusion_matrix.
    """
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(
            y_test, y_pred, target_names=target_names, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }


def save_model(model: RandomForestClassifier, filepath: str) -> None:
    """Persist a trained model to disk using pickle."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(model, f)


def load_model(filepath: str) -> RandomForestClassifier:
    """Load a persisted model from disk."""
    with open(filepath, "rb") as f:
        return pickle.load(f)


def train_and_evaluate(data: dict) -> dict:
    """
    Convenience wrapper: train both models and return evaluation metrics.

    Args:
        data: output dict from preprocessing.preprocess()

    Returns:
        dict with keys: diag_model, status_model, diag_metrics, status_metrics
    """
    diag_model = train_diagnosis_model(data["X_train"], data["y_diag_train"])
    status_model = train_status_model(data["X_train"], data["y_status_train"])

    diag_metrics = evaluate_model(
        diag_model, data["X_test"], data["y_diag_test"], DIAGNOSIS_LABELS
    )
    status_metrics = evaluate_model(
        status_model, data["X_test"], data["y_status_test"], STATUS_LABELS
    )

    return {
        "diag_model": diag_model,
        "status_model": status_model,
        "diag_metrics": diag_metrics,
        "status_metrics": status_metrics,
    }
