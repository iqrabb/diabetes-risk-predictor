# 🩺 Diabetes Prediction System

A web-based Machine Learning application that predicts the risk of diabetes in a patient based on health parameters, using an **XGBoost Classifier** and an interactive **Streamlit** interface.

---

## 📌 Overview

This project takes patient health data (age, BMI, glucose level, HbA1c, hypertension, heart disease, smoking history, etc.) as input and predicts whether the patient is at **high risk** or **low risk** of diabetes, along with the prediction probability/confidence.

---

## 🚀 Features

- Clean and interactive **Streamlit UI**
- Real-time diabetes risk prediction
- Displays prediction probability (Diabetic vs Non-Diabetic %)
- Patient data summary table
- Model performance metrics shown in sidebar
- Fast and lightweight (JSON-based model loading)

---

## 🧠 Model Details

- **Algorithm:** XGBoost Classifier
- **Model format:** `.json` (XGBoost native format — version-independent and reliable)

### 📊 Performance Metrics

| Metric     | Score   |
|------------|---------|
| Accuracy   | 95.6%   |
| Precision  | 73%     |
| Recall     | 81%     |
| ROC-AUC    | 97.92%  |

> Note: `scale_pos_weight=4` was used during training to handle class imbalance (diabetic cases were ~10x fewer than non-diabetic cases in the dataset), which significantly improved recall for detecting actual diabetes cases.

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit** – Web app interface
- **XGBoost** – Machine learning model
- **Pandas** – Data handling
- **Scikit-learn** – Preprocessing / evaluation

---

## 📂 Project Structure

```
Diabetes_Prediction/
├── app.py                  # Main Streamlit application
├── diabetes_model.json     # Trained XGBoost model
├── requirements.txt        # Python dependencies
├── .gitignore
└── README.md
```

---

## ⚙️ Installation & Setup

1. **Clone the repository**
```bash
git clone <your-repo-link>
cd Diabetes_Prediction
```

2. **Create a virtual environment (recommended)**
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the app**
```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

---

## 📋 Input Parameters

| Feature              | Description                              |
|-----------------------|-------------------------------------------|
| Gender               | Female / Male / Other                    |
| Age                  | Patient age (1–100)                      |
| Hypertension         | 0 = No, 1 = Yes                          |
| Heart Disease        | 0 = No, 1 = Yes                          |
| BMI                  | Body Mass Index                          |
| HbA1c Level          | Blood sugar (average) level              |
| Blood Glucose Level  | Current blood glucose level              |
| Smoking History      | Never / Former / Current / Ever / Not Current |

---

## 📈 Output

- **Prediction:** High Risk ⚠️ / Low Risk ✅
- **Diabetes Probability (%)**
- **Non-Diabetes Probability (%)**
- **Prediction Confidence Bar**
- **Patient Summary Table**

---

## 👤 Author

Developed by **Iqra Bibi** ❤️

---

## 📄 License

This project is open-source and available for educational and personal use.