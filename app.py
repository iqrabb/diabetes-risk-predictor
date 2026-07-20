import streamlit as st
import pandas as pd
import xgboost as xgb

# ===============================
# PAGE CONFIGURATION
# ===============================
st.set_page_config(
    page_title="Diabetes Prediction System",
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
# CUSTOM CSS
# ===============================
st.markdown("""
<style>

.main{
    background-color:#F5F9FF;
}

.title{
    font-size:42px;
    font-weight:bold;
    color:#0B5ED7;
}

.subtitle{
    color:#555;
    font-size:18px;
}

.metric-box{
    background:white;
    padding:18px;
    border-radius:15px;
    text-align:center;
    box-shadow:0px 3px 12px rgba(0,0,0,.08);
}

.result-success{
    background:#D1FAE5;
    padding:20px;
    border-radius:15px;
    color:#065F46;
    font-size:24px;
    font-weight:bold;
}

.result-danger{
    background:#FEE2E2;
    padding:20px;
    border-radius:15px;
    color:#991B1B;
    font-size:24px;
    font-weight:bold;
}

.footer{
    text-align:center;
    color:gray;
    font-size:15px;
}

</style>
""", unsafe_allow_html=True)

# ===============================
# HEADER
# ===============================

st.markdown("<div class='title'>🩺 Diabetes Prediction System</div>", unsafe_allow_html=True)

st.markdown(
"<div class='subtitle'>Predict Diabetes Risk using an XGBoost Machine Learning Model</div>",
unsafe_allow_html=True
)

st.write("")

# ===============================
# SIDEBAR
# ===============================

st.sidebar.title("🩺 Project Information")

st.sidebar.success("""
### Model

✅ XGBoost Classifier

### Performance

✔ Accuracy : 95.6%

✔ Precision : 73%

✔ Recall : 81%

✔ ROC-AUC : 97.92%

""")

st.sidebar.info(
"""
Developed using

- Python
- Streamlit
- XGBoost
- Scikit-learn
"""
)

# ===============================
# INPUT SECTION
# ===============================

st.header("📋 Patient Information")

col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Female","Male","Other"]
    )

    age = st.slider(
        "Age",
        1,
        100,
        35
    )

    hypertension = st.selectbox(
        "Hypertension",
        [0,1]
    )

    heart_disease = st.selectbox(
        "Heart Disease",
        [0,1]
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
        [
            "never",
            "former",
            "current",
            "ever",
            "not current"
        ]
    )

st.write("")
st.write("")

# ===============================
# PREDICTION
# ===============================

if st.button("🔍 Predict Diabetes Risk", use_container_width=True):

    input_df = pd.DataFrame({

        "age":[age],

        "hypertension":[hypertension],

        "heart_disease":[heart_disease],

        "bmi":[bmi],

        "HbA1c_level":[hba1c],

        "blood_glucose_level":[glucose],

        "gender_Male":[1 if gender=="Male" else 0],

        "gender_Other":[1 if gender=="Other" else 0],

        "smoking_history_current":[1 if smoking=="current" else 0],

        "smoking_history_ever":[1 if smoking=="ever" else 0],

        "smoking_history_former":[1 if smoking=="former" else 0],

        "smoking_history_never":[1 if smoking=="never" else 0],

        "smoking_history_not current":[1 if smoking=="not current" else 0]

    })

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0]

    diabetic = probability[1]*100

    non_diabetic = probability[0]*100

    st.divider()

    st.subheader("📊 Prediction Result")

    if prediction==1:

        st.markdown(
        f"""
<div class="result-danger">
⚠️ High Risk of Diabetes
</div>
""",
unsafe_allow_html=True)

    else:

        st.markdown(
        f"""
<div class="result-success">
✅ Low Risk of Diabetes
</div>
""",
unsafe_allow_html=True)

    st.write("")

    c1,c2=st.columns(2)

    with c1:

        st.metric(
            "Diabetes Probability",
            f"{diabetic:.2f}%"
        )

    with c2:

        st.metric(
            "Non-Diabetes Probability",
            f"{non_diabetic:.2f}%"
        )

    st.write("### 📈 Prediction Confidence")

    st.progress(float(probability[1]))

    st.write("")

    st.subheader("🧾 Patient Summary")

    summary = pd.DataFrame({

        "Feature":[
            "Age",
            "Gender",
            "BMI",
            "HbA1c",
            "Blood Glucose",
            "Hypertension",
            "Heart Disease",
            "Smoking History"
        ],

        "Value":[
            age,
            gender,
            bmi,
            hba1c,
            glucose,
            hypertension,
            heart_disease,
            smoking
        ]

    })

    st.dataframe(summary, use_container_width=True)

# ===============================
# FOOTER
# ===============================

st.write("")
st.write("")
st.markdown("---")

st.markdown(
"""
<div class="footer">

Developed by <b>Iqra Bibi</b> ❤️ <br>

</div>
""",
unsafe_allow_html=True
)