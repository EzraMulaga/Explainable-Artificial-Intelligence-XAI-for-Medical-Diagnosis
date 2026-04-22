# Black-Box AI + XAI for Medical Diagnosis (SHAP Notebook Results)

Final year paper for CS410 — Research and Presentation Skills.

This repository is now notebook-first and is focused on one core research
question: how to integrate a **black-box AI model** with **XAI techniques** to
understand model decision making.

The primary artifact is:
- `notebooks/xai_demo.ipynb` — end-to-end SHAP analysis notebook

---

## Research Scope

Clinical AI systems often achieve strong predictive performance but remain hard
to trust because they behave as black boxes. This project focuses on integrating
a black-box model with SHAP (SHapley Additive exPlanations) so the prediction
process can be inspected, explained, and discussed in the research paper.

---

## Notebook Workflow

```
Patient dataset (CSV)
        │
        ▼
Notebook executes preprocessing + model training
        │
        ▼
Notebook computes SHAP explanations
        │
        ▼
Notebook presents metrics + interpretation of model decisions
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
│── overview/
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

### Option A — Interactive notebook (primary)

```bash
jupyter notebook notebooks/xai_demo.ipynb
```

### Option B — Regenerate notebook outputs non-interactively

```bash
jupyter nbconvert --to notebook --execute notebooks/xai_demo.ipynb --output xai_demo.ipynb
```

### Option C — Use source modules directly (optional)

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

- **[Project Overview](overview/project_overview.md)** — detailed description of every file and design decision
- **[Research Paper](overview/research_paper.md)** — paper content aligned to black-box + XAI integration

---

## License

This project is released under the MIT License.
