import streamlit as st
import pandas as pd
import xgboost as xgb

# ===============================
# PAGE CONFIGURATION
# ===============================
st.set_page_config(
    page_title="Diabetes Risk Assessment",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================
# LOAD MODEL
# ===============================

model = xgb.XGBClassifier()
model.load_model("diabetes_model.json")

# ===============================
# THEME: CLINICAL TEAL
# ===============================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ---------- App background ---------- */
.stApp {
    background: #F8FAFB;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F766E 0%, #0B5C56 100%);
}

section[data-testid="stSidebar"] * {
    color: #E6FFFA !important;
}

section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.15);
}

/* ---------- Header ---------- */
.hero {
    padding: 6px 0 28px 0;
    border-bottom: 1px solid #E2E8F0;
    margin-bottom: 32px;
}

.eyebrow {
    font-family: 'Manrope', sans-serif;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #0891B2;
    margin-bottom: 6px;
}

.title {
    font-family: 'Manrope', sans-serif;
    font-size: 40px;
    font-weight: 800;
    color: #0F172A;
    letter-spacing: -0.02em;
    line-height: 1.15;
}

.subtitle {
    color: #64748B;
    font-size: 16px;
    margin-top: 8px;
    font-weight: 400;
}

/* ---------- Section headers ---------- */
.section-label {
    font-family: 'Manrope', sans-serif;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #0F766E;
    margin-bottom: 4px;
}

.section-title {
    font-family: 'Manrope', sans-serif;
    font-size: 22px;
    font-weight: 700;
    color: #0F172A;
    margin-bottom: 20px;
}

/* ---------- Input card ---------- */
.input-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 28px 32px;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    margin-bottom: 24px;
}

/* ---------- Streamlit widget label tweaks ---------- */
.stSelectbox label, .stSlider label, .stNumberInput label {
    font-family: 'Manrope', sans-serif !important;
    font-weight: 600 !important;
    font-size: 13.5px !important;
    color: #334155 !important;
}

/* ---------- Predict button ---------- */
.stButton > button {
    background: #0F766E !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 14px 0 !important;
    font-family: 'Manrope', sans-serif !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    letter-spacing: 0.01em;
    transition: all 0.15s ease;
    box-shadow: 0 4px 14px rgba(15, 118, 110, 0.25);
}

.stButton > button:hover {
    background: #0B5C56 !important;
    box-shadow: 0 6px 18px rgba(15, 118, 110, 0.35);
    transform: translateY(-1px);
}

/* ---------- Result cards ---------- */
.result-card {
    padding: 28px 32px;
    border-radius: 16px;
    display: flex;
    align-items: center;
    gap: 18px;
    margin-bottom: 8px;
}

.result-danger {
    background: #FEF2F2;
    border: 1px solid #FECACA;
}

.result-success {
    background: #ECFDF5;
    border: 1px solid #A7F3D0;
}

.result-icon {
    font-size: 40px;
    line-height: 1;
}

.result-text-main {
    font-family: 'Manrope', sans-serif;
    font-size: 22px;
    font-weight: 800;
}

.result-danger .result-text-main { color: #991B1B; }
.result-success .result-text-main { color: #065F46; }

.result-text-sub {
    font-size: 14px;
    color: #64748B;
    margin-top: 2px;
}

/* ---------- Metric cards ---------- */
.metric-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 20px 24px;
    text-align: center;
}

.metric-label {
    font-size: 13px;
    color: #64748B;
    font-weight: 600;
    margin-bottom: 6px;
}

.metric-value {
    font-family: 'Manrope', sans-serif;
    font-size: 30px;
    font-weight: 800;
    color: #0F172A;
}

.metric-value.risk { color: #DC2626; }
.metric-value.safe { color: #0F766E; }

/* ---------- Disclaimer ---------- */
.disclaimer {
    background: #FFFBEB;
    border: 1px solid #FDE68A;
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 13.5px;
    color: #92400E;
    margin-top: 20px;
}

/* ---------- Footer ---------- */
.footer {
    text-align: center;
    color: #94A3B8;
    font-size: 13.5px;
    padding-top: 8px;
}

/* Hide default streamlit chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ===============================
# SIDEBAR
# ===============================

st.sidebar.markdown("### 🩺 Diabetes Risk Assessment")
st.sidebar.markdown(
    "This tool estimates diabetes risk based on standard clinical indicators "
    "such as glucose level, BMI, and HbA1c."
)

st.sidebar.markdown("---")

st.sidebar.markdown("#### How it works")
st.sidebar.markdown(
    "1. Enter the patient's health details on the right\n"
    "2. Click **Predict Diabetes Risk**\n"
    "3. Review the risk result and probability"
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    "<div style='font-size:12.5px; opacity:0.85;'>"
    "This tool provides a statistical risk estimate and is not a medical diagnosis. "
    "Please consult a healthcare professional for medical advice."
    "</div>",
    unsafe_allow_html=True
)

# ===============================
# HEADER
# ===============================

st.markdown("""
<div class="hero">
    <div class="eyebrow">Health Risk Screening</div>
    <div class="title">🩺 Diabetes Risk Assessment</div>
    <div class="subtitle">Enter patient details below to estimate diabetes risk in seconds.</div>
</div>
""", unsafe_allow_html=True)

# ===============================
# INPUT SECTION
# ===============================

st.markdown('<div class="section-label">Step 1</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Patient Information</div>', unsafe_allow_html=True)

st.markdown('<div class="input-card">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Female", "Male", "Other"]
    )

    age = st.slider(
        "Age",
        1,
        100,
        35
    )

    hypertension = st.selectbox(
        "Hypertension",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    heart_disease = st.selectbox(
        "Heart Disease",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

with col2:

    bmi = st.number_input(
        "BMI",
        10.0,
        70.0,
        25.5
    )

    hba1c = st.number_input(
        "HbA1c Level",
        3.0,
        15.0,
        5.8
    )

    glucose = st.number_input(
        "Blood Glucose Level",
        50,
        350,
        120
    )

    smoking = st.selectbox(
        "Smoking History",
        ["never", "former", "current", "ever", "not current"]
    )

st.markdown('</div>', unsafe_allow_html=True)

predict_clicked = st.button("🔍 Predict Diabetes Risk", use_container_width=True)

# ===============================
# PREDICTION
# ===============================

if predict_clicked:

    input_df = pd.DataFrame({

        "age": [age],
        "hypertension": [hypertension],
        "heart_disease": [heart_disease],
        "bmi": [bmi],
        "HbA1c_level": [hba1c],
        "blood_glucose_level": [glucose],
        "gender_Male": [1 if gender == "Male" else 0],
        "gender_Other": [1 if gender == "Other" else 0],
        "smoking_history_current": [1 if smoking == "current" else 0],
        "smoking_history_ever": [1 if smoking == "ever" else 0],
        "smoking_history_former": [1 if smoking == "former" else 0],
        "smoking_history_never": [1 if smoking == "never" else 0],
        "smoking_history_not current": [1 if smoking == "not current" else 0]

    })

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    diabetic = probability[1] * 100
    non_diabetic = probability[0] * 100

    st.write("")
    st.markdown('<div class="section-label">Step 2</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Result</div>', unsafe_allow_html=True)

    if prediction == 1:
        st.markdown("""
        <div class="result-card result-danger">
            <div class="result-icon">⚠️</div>
            <div>
                <div class="result-text-main">High Risk of Diabetes</div>
                <div class="result-text-sub">Consider consulting a healthcare professional for further evaluation.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-card result-success">
            <div class="result-icon">✅</div>
            <div>
                <div class="result-text-main">Low Risk of Diabetes</div>
                <div class="result-text-sub">No strong risk indicators detected based on the provided data.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Diabetes Probability</div>
            <div class="metric-value risk">{diabetic:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Non-Diabetes Probability</div>
            <div class="metric-value safe">{non_diabetic:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.progress(float(probability[1]))

    st.write("")
    st.markdown('<div class="section-label">Summary</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Patient Summary</div>', unsafe_allow_html=True)

    summary = pd.DataFrame({
        "Feature": [
            "Age", "Gender", "BMI", "HbA1c", "Blood Glucose",
            "Hypertension", "Heart Disease", "Smoking History"
        ],
        "Value": [
            age, gender, bmi, hba1c, glucose,
            "Yes" if hypertension == 1 else "No",
            "Yes" if heart_disease == 1 else "No",
            smoking
        ]
    })

    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="disclaimer">
        ⚠️ This result is a statistical estimate based on machine learning and is <b>not a medical diagnosis</b>.
        Please consult a qualified healthcare professional for accurate diagnosis and treatment.
    </div>
    """, unsafe_allow_html=True)

# ===============================
# FOOTER
# ===============================

st.write("")
st.write("")
st.markdown("---")
st.markdown('<div class="footer">Developed by <b>Iqra Bibi</b> ❤️</div>', unsafe_allow_html=True)
