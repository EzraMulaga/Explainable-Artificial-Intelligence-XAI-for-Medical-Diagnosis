"""
preprocessing.py
----------------
Handles data loading, missing value imputation, and feature normalization
for the XAI Medical Diagnosis pipeline.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


FEATURE_COLUMNS = ["heart_rate", "blood_pressure", "spo2", "temperature"]
DIAGNOSIS_COLUMN = "diagnosis"
STATUS_COLUMN = "status"
STATUS_LABELS = ["Stable", "At Risk", "Critical"]


def load_data(filepath: str) -> pd.DataFrame:
    """Load patient data from a CSV file."""
    df = pd.read_csv(filepath)
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing numeric values with the column median and
    fill missing categorical values with the most frequent value.
    """
    df = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].mode()[0], inplace=True)
    return df


def encode_status(df: pd.DataFrame) -> pd.DataFrame:
    """Encode the patient status column as an integer label."""
    df = df.copy()
    status_map = {label: idx for idx, label in enumerate(STATUS_LABELS)}
    df[STATUS_COLUMN] = df[STATUS_COLUMN].map(status_map)
    return df


def normalize_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """
    Fit a MinMaxScaler on training features and transform both splits.

    Returns:
        X_train_scaled, X_test_scaled, fitted scaler
    """
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns
    )
    return X_train_scaled, X_test_scaled, scaler


def preprocess(
    filepath: str, test_size: float = 0.2, random_state: int = 42
) -> dict:
    """
    Full preprocessing pipeline.

    Returns a dict with keys:
        X_train, X_test, y_diag_train, y_diag_test,
        y_status_train, y_status_test, scaler, feature_names
    """
    df = load_data(filepath)
    df = handle_missing_values(df)
    df = encode_status(df)

    X = df[FEATURE_COLUMNS]
    y_diagnosis = df[DIAGNOSIS_COLUMN]
    y_status = df[STATUS_COLUMN]

    X_train, X_test, y_diag_train, y_diag_test, y_status_train, y_status_test = (
        train_test_split(
            X,
            y_diagnosis,
            y_status,
            test_size=test_size,
            random_state=random_state,
            stratify=y_diagnosis,
        )
    )

    X_train_scaled, X_test_scaled, scaler = normalize_features(X_train, X_test)

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_diag_train": y_diag_train.reset_index(drop=True),
        "y_diag_test": y_diag_test.reset_index(drop=True),
        "y_status_train": y_status_train.reset_index(drop=True),
        "y_status_test": y_status_test.reset_index(drop=True),
        "scaler": scaler,
        "feature_names": FEATURE_COLUMNS,
    }
