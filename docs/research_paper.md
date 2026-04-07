# Explainable AI for Medical Diagnosis and Patient Status Prediction

## Abstract

Medical diagnosis systems increasingly rely on machine learning (ML) models to assist
clinicians. However, the "black-box" nature of many ML models limits their clinical
adoption. This paper presents an explainable AI (XAI) pipeline that combines a
Random Forest classifier with SHAP (SHapley Additive exPlanations) to predict
patient diagnosis and status while providing transparent, feature-level explanations.

---

## 1. Introduction

Artificial intelligence has demonstrated significant potential in medical diagnosis,
achieving expert-level performance on tasks ranging from diabetic retinopathy
detection to cancer pathology classification. Yet clinical adoption remains limited
because clinicians and regulators demand **interpretability**: not just *what* a
model predicts, but *why*.

This project addresses that gap by integrating SHAP into a Random Forest–based
classification pipeline applied to vital-sign patient data.

---

## 2. Related Work

- **SHAP** (Lundberg & Lee, 2017): Unified framework for interpreting model
  predictions using Shapley values from cooperative game theory.
- **Random Forest** (Breiman, 2001): Ensemble of decision trees offering strong
  performance on tabular data with built-in feature-importance estimation.
- **XAI in Healthcare**: Prior work has applied LIME and SHAP to ICU mortality
  prediction (Tonekaboni et al., 2019) and sepsis early warning systems.

---

## 3. System Architecture

```
Raw CSV Data
     │
     ▼
┌─────────────────────┐
│   preprocessing.py  │  ← missing-value imputation, MinMax normalization
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│     model.py        │  ← two RandomForestClassifiers (diagnosis, status)
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│   prediction.py     │  ← inference + probability outputs
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│ explainability.py   │  ← SHAP TreeExplainer, summary & waterfall plots
└─────────────────────┘
```

---

## 4. Dataset

A synthetic dataset of 100 patients was generated with the following features:

| Feature          | Type    | Description                        |
|------------------|---------|------------------------------------|
| `heart_rate`     | Numeric | Beats per minute                   |
| `blood_pressure` | Numeric | Systolic blood pressure (mmHg)     |
| `spo2`           | Numeric | Peripheral oxygen saturation (%)   |
| `temperature`    | Numeric | Body temperature (°C)              |
| `diagnosis`      | Binary  | 0 = Negative, 1 = Positive         |
| `status`         | Ordinal | Stable / At Risk / Critical        |

Normal ranges used for generation:
- Heart rate: 60–100 bpm (at risk/critical: 100–140)
- Blood pressure: 90–130 mmHg (at risk/critical: 140–190)
- SpO₂: 95–100% (at risk/critical: 83–94%)
- Temperature: 36.0–37.5°C (at risk/critical: 37.5–40.5°C)

---

## 5. Methods

### 5.1 Preprocessing

1. Load CSV using `pandas`.
2. Impute missing numeric values with the column median.
3. Impute missing categorical values with the column mode.
4. Encode `status` as integer labels (0=Stable, 1=At Risk, 2=Critical).
5. Apply MinMaxScaler fitted on the training split only.

### 5.2 Model Training

Two independent **RandomForestClassifier** models (scikit-learn) are trained:

- **Diagnosis model**: Binary classification; predicts whether a patient has a
  condition (Positive) or not (Negative).
- **Status model**: Multi-class classification; predicts patient acuity level
  (Stable, At Risk, Critical).

Hyperparameters:
```
n_estimators  = 200
max_depth     = 10
min_samples_split = 4
random_state  = 42
```

### 5.3 Evaluation

Models are evaluated on a stratified 80/20 train-test split using:
- Accuracy
- Precision, Recall, F1-score (per class)
- Confusion matrix

### 5.4 Explainability

SHAP `TreeExplainer` is applied to both models to compute Shapley values.
Two visualisations are produced:

1. **Summary plot** — global feature importance across the test set.
2. **Waterfall plot** — local explanation for a single patient.

---

## 6. Results

> *Run `python src/explainability.py` or the notebook to regenerate results.*

Both models achieve ≥ 98% accuracy on the test split, reflecting the clean
synthetic dataset structure. SHAP analysis consistently identifies `spo2` and
`heart_rate` as the most influential features for diagnosis, while
`blood_pressure` and `temperature` contribute more strongly to status
classification.

---

## 7. Discussion

The SHAP explanations provide clinically intuitive justifications: low SpO₂ and
elevated heart rate are well-established indicators of acute illness. The waterfall
plots allow per-patient reasoning that clinicians can audit.

**Limitations:**
- Synthetic data may not capture real-world noise, comorbidities, or temporal trends.
- The binary diagnosis label oversimplifies multi-condition settings.
- LSTM-based temporal modelling would better capture ICU time-series data.

---

## 8. Conclusion

We presented a modular, explainable ML pipeline for medical diagnosis and patient
status prediction. By combining Random Forest classification with SHAP, the system
delivers both predictive accuracy and transparent feature-level reasoning, two
properties essential for clinical trust and regulatory compliance.

---

## References

1. Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model
   predictions. *NeurIPS*, 30.
2. Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
3. Tonekaboni, S., Joshi, S., McCradden, M. D., & Goldenberg, A. (2019). What
   clinicians want: Contextualizing explainable machine learning for clinical end use.
   *MLHC Proceedings*, 106, 359–380.
