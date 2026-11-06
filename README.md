# Diabetes Risk Predictor

An interactive Streamlit application that uses a saved XGBoost classifier to estimate diabetes probability from health-related inputs. The app presents a probability estimate, a simple risk band, a factor-by-factor summary, and a downloadable text report.

> **Important:** This is an educational machine-learning demonstration, not a medical device. It does not diagnose diabetes, replace laboratory testing, or provide medical advice. Do not use its output to make healthcare decisions.

## Features

- Streamlit interface for entering patient-related features.
- XGBoost model loaded from the native `diabetes_model.json` format.
- Probability output for diabetes and non-diabetes classes.
- App-defined Low, Moderate, and High risk bands.
- Rule-based display of selected input values.
- Downloadable text summary of the entered values and model output.

## Tech stack

- Python
- Streamlit
- XGBoost
- Pandas
- scikit-learn (listed in the project dependencies)

## How the application works

1. The app loads `diabetes_model.json` using XGBoost.
2. The user enters age, gender, BMI, HbA1c, blood glucose, hypertension, heart disease, and smoking history.
3. The app converts categorical inputs into the feature columns expected by the model.
4. XGBoost returns class probabilities.
5. The app displays the estimated probability and maps it to a risk band using fixed thresholds in the application code.

The risk bands are application-defined thresholds (below 30%, 30–60%, and 60% or above). They have **not been established as clinically validated decision thresholds**.

## Run locally

### 1. Clone the repository

```bash
git clone https://github.com/iqrabb/diabetes-risk-predictor.git
cd diabetes-risk-predictor
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start Streamlit

```bash
streamlit run app.py
```

Streamlit will print a local URL, usually http://localhost:8501.

The app expects `diabetes_model.json` to be in the repository root alongside `app.py`.

## Input features

| Input | Description |
|---|---|
| Gender | Female, Male, or Other |
| Age | Age in years |
| Hypertension | No / Yes |
| Heart disease | No / Yes |
| BMI | Body mass index |
| HbA1c level | HbA1c value entered by the user |
| Blood glucose level | Blood glucose value in mg/dL |
| Smoking history | Never, former, current, ever, or not current |

## Repository contents

- `app.py` — Streamlit application and input/prediction logic.
- `diabetes_model.json` — saved XGBoost model.
- `diabetes_prediction_dataset.csv` — dataset file included in the repository.
- `requirements.txt` — Python dependencies.
- `README.md` — project documentation.

## Model evaluation and reproducibility

The repository includes a dataset and a saved model, but it currently does not include a training script or notebook that reproduces the model. The performance figures previously shown in this README have therefore been removed from the project overview: the evaluation procedure and results are not independently verifiable from the committed files alone.

For a reproducible project, add the training pipeline, document the dataset source and license, define a leakage-safe train/validation/test strategy, and report class-specific precision, recall, F1-score, confusion matrix, and ROC-AUC on a held-out test set.

## Limitations and responsible use

- Model quality depends on the provenance, representativeness, and quality of the training data.
- The probability output should not be interpreted as a calibrated personal risk.
- The risk bands and factor summaries in the UI are simplified application logic, not validated clinical guidance.
- The app does not verify laboratory results or patient history.
- Consult a qualified healthcare professional for screening, diagnosis, and treatment decisions.

---

Developed as an educational demonstration of tabular machine learning and Streamlit application development.
