# Explainable AI for Medical Diagnosis and Patient Status Prediction

Final year paper for CS410 — Research and Presentation Skills

A modular, research-grade machine learning system that predicts patient
**diagnosis** and **acuity status** from vital-sign data, while providing
transparent feature-level explanations using **SHAP**.

---

## Problem Statement

Clinical decision-support tools are only trusted when clinicians can understand
*why* a model reaches a conclusion. This project combines a Random Forest
classifier with SHAP (SHapley Additive exPlanations) to deliver both predictive
accuracy and interpretable reasoning for medical diagnosis.

---

## System Pipeline

```
Raw patient vitals (CSV)
        │
        ▼
┌─────────────────┐
│ preprocessing.py│  ← impute missing values, MinMax scale
└─────────────────┘
        │
        ▼
┌─────────────────┐
│   model.py      │  ← train RandomForest for diagnosis + status
└─────────────────┘
        │
        ▼
┌─────────────────┐
│  prediction.py  │  ← output diagnosis, status, class probabilities
└─────────────────┘
        │
        ▼
┌──────────────────────┐
│  explainability.py   │  ← SHAP summary + waterfall plots
└──────────────────────┘
```

---

## Repository Structure

```
project-root/
│── data/
│   └── sample_patient_data.csv   # 100-patient synthetic dataset
│
│── src/
│   ├── preprocessing.py          # data loading, imputation, normalization
│   ├── model.py                  # RandomForest training & evaluation
│   ├── prediction.py             # single-patient & batch inference
│   └── explainability.py         # SHAP explanations & plots
│
│── notebooks/
│   └── xai_demo.ipynb            # end-to-end interactive demo
│
│── docs/
│   ├── research_paper.md         # academic write-up
│   └── project_overview.md       # detailed file & design documentation
│
│── results/
│   ├── metrics.txt               # evaluation metrics (generated at runtime)
│   └── shap_plots.png            # SHAP summary plot (generated at runtime)
│
│── README.md
```

---

## Input Features

| Feature          | Description                        |
|------------------|------------------------------------|
| `heart_rate`     | Beats per minute                   |
| `blood_pressure` | Systolic blood pressure (mmHg)     |
| `spo2`           | Peripheral oxygen saturation (%)   |
| `temperature`    | Body temperature (°C)              |

## Outputs

| Output              | Type                             |
|---------------------|----------------------------------|
| `diagnosis`         | Negative / Positive              |
| `status`            | Stable / At Risk / Critical      |
| `diagnosis_proba`   | Per-class probabilities          |
| `status_proba`      | Per-class probabilities          |
| SHAP summary plot   | Global feature importance chart  |
| SHAP waterfall plot | Per-patient explanation          |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/EzraMulaga/Explainable-Artificial-Intelligence-XAI-for-Medical-Diagnosis.git
cd Explainable-Artificial-Intelligence-XAI-for-Medical-Diagnosis

# Install Python dependencies
pip install pandas numpy scikit-learn shap matplotlib jupyter
```

---

## How to Run

### Option A — Interactive notebook (recommended)

```bash
jupyter notebook notebooks/xai_demo.ipynb
```

### Option B — Command-line scripts

```bash
# Run full pipeline: preprocess → train → evaluate → SHAP plots
cd src
python explainability.py
```

```bash
# Run single-patient prediction demo
cd src
python prediction.py
```

### Option C — Import as a library

```python
from src.preprocessing import preprocess
from src.model import train_and_evaluate
from src.prediction import predict_single, format_prediction_report
from src.explainability import compute_shap_values, plot_summary

data    = preprocess("data/sample_patient_data.csv")
results = train_and_evaluate(data)

patient = {"heart_rate": 110, "blood_pressure": 155, "spo2": 91, "temperature": 38.4}
pred    = predict_single(results["diag_model"], results["status_model"], data["scaler"], patient)
print(format_prediction_report(pred))
```

---

## Documentation

- **[Project Overview](docs/project_overview.md)** — detailed description of every file and design decision
- **[Research Paper](docs/research_paper.md)** — academic write-up with methodology and references

---

## License

This project is released under the MIT License.
