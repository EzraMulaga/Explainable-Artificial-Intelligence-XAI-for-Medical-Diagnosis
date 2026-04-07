"""
explainability.py
-----------------
Uses SHAP (SHapley Additive exPlanations) to explain predictions made by the
trained RandomForest classifiers.

Provides:
  - compute_shap_values()    : compute SHAP values for a dataset
  - plot_summary()           : generate and optionally save a SHAP summary plot
  - explain_single()         : text explanation for a single patient prediction
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server/CI environments
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier


def compute_shap_values(
    model: RandomForestClassifier,
    X: pd.DataFrame,
) -> shap.Explanation:
    """
    Compute SHAP values for a dataset using TreeExplainer.

    Args:
        model: trained RandomForestClassifier
        X:     feature DataFrame (scaled)

    Returns:
        shap.Explanation object
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    return shap_values


def plot_summary(
    shap_values: shap.Explanation,
    X: pd.DataFrame,
    title: str = "SHAP Feature Importance",
    save_path: str = None,
    class_index: int = 1,
) -> None:
    """
    Generate a SHAP beeswarm/summary plot.

    Args:
        shap_values: output of compute_shap_values()
        X:           feature DataFrame used for axis labels
        title:       plot title
        save_path:   if provided, save the figure to this path
        class_index: for multi-class models, the class index to plot
    """
    plt.figure(figsize=(10, 6))

    sv = shap_values
    # For multi-output models, select the relevant class slice
    if sv.values.ndim == 3:
        sv_plot = shap.Explanation(
            values=sv.values[:, :, class_index],
            base_values=sv.base_values[:, class_index]
            if sv.base_values.ndim > 1
            else sv.base_values,
            data=sv.data,
            feature_names=sv.feature_names,
        )
    else:
        sv_plot = sv

    shap.summary_plot(sv_plot, X, show=False)
    plt.title(title)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"SHAP plot saved to: {save_path}")

    plt.close()


def plot_waterfall(
    shap_values: shap.Explanation,
    sample_index: int = 0,
    title: str = "SHAP Waterfall – Single Patient",
    save_path: str = None,
    class_index: int = 1,
) -> None:
    """
    Generate a SHAP waterfall plot for a single sample.

    Args:
        shap_values:  output of compute_shap_values()
        sample_index: which row in the dataset to explain
        title:        plot title
        save_path:    optional file path to save the figure
        class_index:  class to explain for multi-class models
    """
    sv = shap_values
    if sv.values.ndim == 3:
        explanation = shap.Explanation(
            values=sv.values[sample_index, :, class_index],
            base_values=sv.base_values[sample_index, class_index]
            if sv.base_values.ndim > 1
            else sv.base_values[sample_index],
            data=sv.data[sample_index],
            feature_names=sv.feature_names,
        )
    else:
        explanation = sv[sample_index]

    plt.figure(figsize=(10, 4))
    shap.waterfall_plot(explanation, show=False)
    plt.title(title)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Waterfall plot saved to: {save_path}")

    plt.close()


def explain_single(
    model: RandomForestClassifier,
    X_single: pd.DataFrame,
    feature_names: list[str],
    top_n: int = 4,
    class_index: int = 1,
) -> str:
    """
    Return a text explanation of which features most influenced a prediction.

    Args:
        model:         trained RandomForestClassifier
        X_single:      single-row scaled DataFrame
        feature_names: list of feature column names
        top_n:         number of top features to report
        class_index:   class to explain for multi-class models

    Returns:
        Human-readable explanation string
    """
    explainer = shap.TreeExplainer(model)
    sv = explainer(X_single)

    if sv.values.ndim == 3:
        vals = sv.values[0, :, class_index]
    else:
        vals = sv.values[0]

    contributions = sorted(
        zip(feature_names, vals), key=lambda x: abs(x[1]), reverse=True
    )

    lines = ["Feature contributions (SHAP values):"]
    for feat, val in contributions[:top_n]:
        direction = "↑ increases" if val > 0 else "↓ decreases"
        lines.append(f"  {feat:20s}: {val:+.4f}  ({direction} predicted risk)")

    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    from preprocessing import preprocess
    from model import train_and_evaluate

    DATA_PATH = os.path.join(
        os.path.dirname(__file__), "..", "data", "sample_patient_data.csv"
    )
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

    data = preprocess(DATA_PATH)
    results = train_and_evaluate(data)

    # SHAP for diagnosis model
    shap_vals = compute_shap_values(results["diag_model"], data["X_test"])
    plot_summary(
        shap_vals,
        data["X_test"],
        title="SHAP Summary – Diagnosis Model",
        save_path=os.path.join(RESULTS_DIR, "shap_plots.png"),
        class_index=1,
    )

    # Text explanation for first test patient
    explanation = explain_single(
        results["diag_model"],
        data["X_test"].iloc[[0]],
        data["feature_names"],
    )
    print(explanation)
