# Project Overview — Explainable AI for Medical Diagnosis

This document describes the purpose, structure, and responsibilities of every
file in this project.

---

## Repository Structure

```
project-root/
│── data/
│   └── sample_patient_data.csv
│
│── src/
│   ├── preprocessing.py
│   ├── model.py
│   ├── prediction.py
│   └── explainability.py
│
│── notebooks/
│   └── xai_demo.ipynb
│
│── docs/
│   ├── research_paper.md
│   └── project_overview.md   ← this file
│
│── results/
│   ├── metrics.txt
│   └── shap_plots.png        ← generated at runtime
│
│── README.md
```

---

## File Descriptions

### `data/sample_patient_data.csv`

**Purpose:** Synthetic dataset simulating 100 patient records.

**Columns:**

| Column           | Description                                     |
|------------------|-------------------------------------------------|
| `patient_id`     | Unique identifier                               |
| `heart_rate`     | Heart rate in beats per minute                  |
| `blood_pressure` | Systolic blood pressure in mmHg                 |
| `spo2`           | Peripheral oxygen saturation percentage         |
| `temperature`    | Body temperature in degrees Celsius             |
| `diagnosis`      | Binary label: 0 = Negative, 1 = Positive        |
| `status`         | Acuity label: Stable / At Risk / Critical       |

Data was hand-crafted to reflect realistic clinical thresholds: patients with
elevated heart rate, low SpO₂, and high temperature are labelled "At Risk" or
"Critical".

---

### `src/preprocessing.py`

**Purpose:** Data loading and preparation for model training and inference.

**Key functions:**

| Function                | Description                                                        |
|-------------------------|--------------------------------------------------------------------|
| `load_data(filepath)`   | Read CSV into a pandas DataFrame                                   |
| `handle_missing_values` | Fill numeric gaps with median; categorical gaps with mode          |
| `encode_status`         | Map `status` strings to integer labels (0, 1, 2)                  |
| `normalize_features`    | Fit MinMaxScaler on training data; transform train and test splits |
| `preprocess(filepath)`  | Full pipeline; returns a dict with scaled splits and scaler        |

**Design decisions:**
- Scaler is fitted *only* on the training split to prevent data leakage.
- Returns the fitted `scaler` object so it can be reused at inference time.

---

### `src/model.py`

**Purpose:** Train and evaluate Random Forest classifiers.

**Key functions:**

| Function                  | Description                                                   |
|---------------------------|---------------------------------------------------------------|
| `train_diagnosis_model`   | Fit a binary RandomForestClassifier for diagnosis prediction  |
| `train_status_model`      | Fit a multi-class classifier for patient status (3 classes)   |
| `evaluate_model`          | Return accuracy, classification report, and confusion matrix  |
| `save_model` / `load_model` | Pickle serialisation for model persistence                 |
| `train_and_evaluate`      | Convenience wrapper: train both models and return metrics     |

**Why Random Forest?**
- Handles non-linear relationships in tabular medical data well.
- Provides built-in feature importance (complemented by SHAP).
- Robust to small datasets and requires little hyperparameter tuning.

---

### `src/prediction.py`

**Purpose:** Inference for individual patients or entire batches.

**Key functions:**

| Function                 | Description                                                        |
|--------------------------|--------------------------------------------------------------------|
| `predict_single`         | Run inference on one patient dict; return diagnosis + status       |
| `predict_batch`          | Vectorised inference on a full DataFrame                           |
| `format_prediction_report` | Format a single prediction as a human-readable string           |

**Output fields from `predict_single`:**

```python
{
    "diagnosis": "Positive",
    "status": "At Risk",
    "diagnosis_proba": {"Negative": 0.12, "Positive": 0.88},
    "status_proba": {"Stable": 0.05, "At Risk": 0.80, "Critical": 0.15}
}
```

---

### `src/explainability.py`

**Purpose:** Compute and visualise SHAP explanations for model predictions.

**Key functions:**

| Function            | Description                                                          |
|---------------------|----------------------------------------------------------------------|
| `compute_shap_values` | Run SHAP `TreeExplainer` on a dataset; returns `shap.Explanation` |
| `plot_summary`      | Beeswarm summary plot of global feature importance                   |
| `plot_waterfall`    | Waterfall plot showing a single patient's feature contributions      |
| `explain_single`    | Text summary of top-N contributing features for one patient          |

**SHAP class index convention:**
- For the **diagnosis model**: `class_index=1` corresponds to "Positive".
- For the **status model**: `class_index=2` corresponds to "Critical".

---

### `notebooks/xai_demo.ipynb`

**Purpose:** Interactive end-to-end demonstration of the full pipeline.

**Sections:**
1. **Setup** — install dependencies, import modules
2. **Data exploration** — load dataset, display statistics, visualise distributions
3. **Preprocessing** — run `preprocess()`, inspect scaled data
4. **Model training** — call `train_and_evaluate()`, display metrics tables
5. **Single prediction** — create a sample patient, run `predict_single()`, print report
6. **SHAP explanations** — compute SHAP values, render summary and waterfall plots
7. **Batch prediction** — run `predict_batch()` on the test set

---

### `docs/research_paper.md`

**Purpose:** Academic-style write-up covering motivation, methodology, results,
and references. Suitable as a basis for a course or conference paper submission.

---

### `results/metrics.txt`

**Purpose:** Stores evaluation metrics produced by `model.py` after training.
Regenerated each run by executing `python src/model.py` or running the notebook.

---

### `results/shap_plots.png`

**Purpose:** SHAP summary beeswarm plot saved automatically when
`explainability.py` is executed. Shows which features most strongly influence
the diagnosis prediction across the test set.

---

### `README.md`

**Purpose:** Top-level project introduction for GitHub visitors. Covers:
- Problem statement
- System pipeline overview
- Installation and setup instructions
- How to run training, prediction, and explanation scripts
- How to open the notebook

---

## Dependency Summary

| Package          | Role                                  |
|------------------|---------------------------------------|
| `pandas`         | Data loading and manipulation         |
| `numpy`          | Numerical operations                  |
| `scikit-learn`   | RandomForestClassifier, scaler, metrics |
| `shap`           | SHAP value computation and plotting   |
| `matplotlib`     | Saving SHAP plots to PNG              |
| `jupyter`        | Running the demo notebook             |

Install all dependencies:

```bash
pip install pandas numpy scikit-learn shap matplotlib jupyter
```
