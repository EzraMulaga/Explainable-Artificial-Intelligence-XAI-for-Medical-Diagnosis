"""
prediction.py
-------------
Runs inference for a single patient or a batch of patients.

Outputs:
  - diagnosis  : "Negative" | "Positive"
  - status     : "Stable" | "At Risk" | "Critical"
  - probabilities for each class
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

from preprocessing import FEATURE_COLUMNS

DIAGNOSIS_LABELS = {0: "Negative", 1: "Positive"}
STATUS_LABELS = {0: "Stable", 1: "At Risk", 2: "Critical"}


def predict_single(
    diag_model: RandomForestClassifier,
    status_model: RandomForestClassifier,
    scaler: MinMaxScaler,
    patient_data: dict,
) -> dict:
    """
    Predict diagnosis and status for a single patient.

    Args:
        diag_model:    trained diagnosis classifier
        status_model:  trained status classifier
        scaler:        fitted MinMaxScaler from the training pipeline
        patient_data:  dict with keys: heart_rate, blood_pressure, spo2, temperature

    Returns:
        dict with keys: diagnosis, status, diagnosis_proba, status_proba
    """
    df = pd.DataFrame([patient_data], columns=FEATURE_COLUMNS)
    X_scaled = pd.DataFrame(scaler.transform(df), columns=FEATURE_COLUMNS)

    diag_pred = int(diag_model.predict(X_scaled)[0])
    status_pred = int(status_model.predict(X_scaled)[0])

    diag_proba = diag_model.predict_proba(X_scaled)[0].tolist()
    status_proba = status_model.predict_proba(X_scaled)[0].tolist()

    return {
        "diagnosis": DIAGNOSIS_LABELS[diag_pred],
        "status": STATUS_LABELS[status_pred],
        "diagnosis_proba": {
            label: round(prob, 4)
            for label, prob in zip(DIAGNOSIS_LABELS.values(), diag_proba)
        },
        "status_proba": {
            label: round(prob, 4)
            for label, prob in zip(STATUS_LABELS.values(), status_proba)
        },
    }


def predict_batch(
    diag_model: RandomForestClassifier,
    status_model: RandomForestClassifier,
    scaler: MinMaxScaler,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Predict diagnosis and status for a batch of patients.

    Args:
        diag_model:   trained diagnosis classifier
        status_model: trained status classifier
        scaler:       fitted MinMaxScaler
        df:           DataFrame with columns matching FEATURE_COLUMNS

    Returns:
        Input DataFrame with added columns: predicted_diagnosis, predicted_status
    """
    X = df[FEATURE_COLUMNS].copy()
    X_scaled = pd.DataFrame(scaler.transform(X), columns=FEATURE_COLUMNS)

    diag_preds = diag_model.predict(X_scaled)
    status_preds = status_model.predict(X_scaled)

    result = df.copy().reset_index(drop=True)
    result["predicted_diagnosis"] = [DIAGNOSIS_LABELS[p] for p in diag_preds]
    result["predicted_status"] = [STATUS_LABELS[p] for p in status_preds]
    return result


def format_prediction_report(prediction: dict) -> str:
    """Return a human-readable string from predict_single output."""
    lines = [
        "=" * 40,
        "  PATIENT PREDICTION REPORT",
        "=" * 40,
        f"  Diagnosis : {prediction['diagnosis']}",
        f"  Status    : {prediction['status']}",
        "",
        "  Diagnosis probabilities:",
    ]
    for label, prob in prediction["diagnosis_proba"].items():
        lines.append(f"    {label:10s}: {prob:.2%}")
    lines.append("")
    lines.append("  Status probabilities:")
    for label, prob in prediction["status_proba"].items():
        lines.append(f"    {label:10s}: {prob:.2%}")
    lines.append("=" * 40)
    return "\n".join(lines)


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(__file__))

    from preprocessing import preprocess
    from model import train_and_evaluate

    DATA_PATH = os.path.join(
        os.path.dirname(__file__), "..", "data", "sample_patient_data.csv"
    )

    data = preprocess(DATA_PATH)
    results = train_and_evaluate(data)

    sample_patient = {
        "heart_rate": 110,
        "blood_pressure": 155,
        "spo2": 91,
        "temperature": 38.4,
    }

    prediction = predict_single(
        results["diag_model"],
        results["status_model"],
        data["scaler"],
        sample_patient,
    )
    print(format_prediction_report(prediction))
