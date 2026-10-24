import re
from datetime import datetime

import pandas as pd
import streamlit as st
import xgboost as xgb

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Diabetes Risk Assessment",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =====================================================
# HELPERS
# =====================================================
def html(markup: str) -> None:
    """Render HTML safely: strips leading spaces so Markdown never turns it into a code block."""
    cleaned = re.sub(r"^\s+", "", markup, flags=re.M)
    st.markdown(cleaned, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_model(path: str = "diabetes_model.json") -> xgb.XGBClassifier:
    mdl = xgb.XGBClassifier()
    mdl.load_model(path)
    return mdl


try:
    model = load_model()
except Exception as err:  # clear message instead of a crash screen
    st.error(
        "The model file **diabetes_model.json** could not be loaded. "
        "Place it in the same folder as app.py and restart the app."
    )
    st.caption(f"Technical detail: {err}")
    st.stop()


def risk_level(p: float):
    """Return (label, colour, message) from the diabetes probability (0-1)."""
    if p < 0.30:
        return "Low risk", "#2DD4BF", "No strong risk indicators in the values provided."
    if p < 0.60:
        return "Moderate risk", "#F5B84B", "Some indicators are elevated. A check-up is worth considering."
    return "High risk", "#F4727C", "Several indicators are elevated. Please see a healthcare professional."


def factor_review(age, bmi, hba1c, glucose, hypertension, heart_disease):
    """Rule-based reading of each input against common clinical reference ranges."""
    rows = []

    if hba1c >= 6.5:
        rows.append(("HbA1c", f"{hba1c:.1f}%", "Diabetic range", "bad"))
    elif hba1c >= 5.7:
        rows.append(("HbA1c", f"{hba1c:.1f}%", "Prediabetic range", "warn"))
    else:
        rows.append(("HbA1c", f"{hba1c:.1f}%", "Normal", "ok"))

    if glucose >= 200:
        rows.append(("Blood glucose", f"{glucose} mg/dL", "Very high", "bad"))
    elif glucose >= 126:
        rows.append(("Blood glucose", f"{glucose} mg/dL", "Elevated", "warn"))
    else:
        rows.append(("Blood glucose", f"{glucose} mg/dL", "Within range", "ok"))

    if bmi >= 30:
        rows.append(("BMI", f"{bmi:.1f}", "Obese range", "bad"))
    elif bmi >= 25:
        rows.append(("BMI", f"{bmi:.1f}", "Overweight range", "warn"))
    elif bmi >= 18.5:
        rows.append(("BMI", f"{bmi:.1f}", "Healthy range", "ok"))
    else:
        rows.append(("BMI", f"{bmi:.1f}", "Underweight range", "warn"))

    rows.append(("Hypertension", "Yes" if hypertension else "No", "Adds risk" if hypertension else "None reported",
                 "warn" if hypertension else "ok"))
    rows.append(("Heart disease", "Yes" if heart_disease else "No", "Adds risk" if heart_disease else "None reported",
                 "warn" if heart_disease else "ok"))
    rows.append(("Age", f"{age}", "Risk rises after 45" if age >= 45 else "Below 45",
                 "warn" if age >= 45 else "ok"))
    return rows


def gauge_svg(p: float, colour: str) -> str:
    """Semicircle gauge drawn as plain SVG (no extra libraries needed)."""
    arc_len = 282.74  # pi * radius(90)
    offset = arc_len * (1 - p)
    return f"""
    <svg viewBox="0 0 200 118" class="gauge">
      <path d="M 10 100 A 90 90 0 0 1 190 100" fill="none" stroke="#1E2C3B" stroke-width="14" stroke-linecap="round"/>
      <path d="M 10 100 A 90 90 0 0 1 190 100" fill="none" stroke="{colour}" stroke-width="14" stroke-linecap="round"
            stroke-dasharray="{arc_len}" stroke-dashoffset="{offset:.2f}"/>
      <text x="100" y="86" text-anchor="middle" class="g-num" fill="#E6EDF3">{p*100:.0f}%</text>
      <text x="100" y="108" text-anchor="middle" class="g-cap" fill="#8A9BB0">chance of diabetes</text>
    </svg>
    """


# =====================================================
# THEME (dark clinical)
# =====================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700&family=DM+Sans:wght@400;500;600&display=swap');

:root {
    --ink: #0A1018;
    --panel: #111B27;
    --panel-2: #0E1721;
    --line: #1E2C3B;
    --text: #E6EDF3;
    --muted: #8A9BB0;
    --teal: #2DD4BF;
    --amber: #F5B84B;
    --rose: #F4727C;
}

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--text); }

.stApp {
    background:
        radial-gradient(900px 420px at 85% -10%, rgba(45,212,191,0.10), transparent 60%),
        var(--ink);
}

header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 2.2rem; max-width: 1180px; }

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: var(--panel-2);
    border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }
section[data-testid="stSidebar"] hr { border-color: var(--line); }
.side-title { font-family: 'Sora', sans-serif; font-size: 18px; font-weight: 600; }
.side-note { font-size: 13px; color: var(--muted) !important; line-height: 1.55; }
.side-step { display:flex; gap:10px; margin: 10px 0; font-size: 14px; }
.side-step b { color: var(--teal) !important; font-family: 'Sora', sans-serif; }

/* ---------- Hero ---------- */
.hero { padding: 4px 0 26px 0; margin-bottom: 8px; }
.hero-badge {
    display:inline-block; font-size: 13px; font-weight: 500; color: var(--teal);
    background: rgba(45,212,191,0.10); border: 1px solid rgba(45,212,191,0.28);
    padding: 5px 12px; border-radius: 999px; margin-bottom: 14px;
}
.hero-title {
    font-family: 'Sora', sans-serif; font-weight: 700; font-size: 44px; line-height: 1.12;
    letter-spacing: -0.03em; color: var(--text);
}
.hero-sub { color: var(--muted); font-size: 17px; margin-top: 10px; max-width: 620px; line-height: 1.55; }

/* ---------- Section titles ---------- */
.sec-title { font-family: 'Sora', sans-serif; font-size: 22px; font-weight: 600; margin: 8px 0 4px 0; }
.sec-sub { color: var(--muted); font-size: 14.5px; margin-bottom: 14px; }

/* ---------- Form card ---------- */
div[data-testid="stForm"] {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 26px 28px 18px 28px;
}

/* ---------- Widgets ---------- */
.stSelectbox label, .stSlider label, .stNumberInput label,
div[data-testid="stWidgetLabel"] p {
    color: var(--muted) !important; font-size: 14px !important; font-weight: 500 !important;
}
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-baseweb="input"] input,
.stNumberInput button {
    background-color: var(--panel-2) !important;
    color: var(--text) !important;
    border-color: var(--line) !important;
}
div[data-baseweb="select"] > div:hover, div[data-baseweb="input"] > div:hover { border-color: var(--teal) !important; }
div[data-baseweb="popover"] ul { background: var(--panel) !important; }
div[data-baseweb="popover"] li { color: var(--text) !important; }
div[data-testid="stSlider"] [role="slider"] { background: var(--teal) !important; box-shadow: none !important; }
div[data-testid="stSlider"] div[data-testid="stThumbValue"] { color: var(--teal) !important; }

/* ---------- Buttons ---------- */
.stButton > button, div[data-testid="stFormSubmitButton"] > button, .stDownloadButton > button {
    border-radius: 12px !important; font-family: 'Sora', sans-serif !important; font-weight: 600 !important;
    font-size: 15.5px !important; padding: 12px 0 !important; transition: transform .15s ease, box-shadow .15s ease;
}
div[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #14B8A6, #0E9F8F) !important; color: #04211E !important; border: none !important;
    box-shadow: 0 8px 24px rgba(20,184,166,0.28);
}
div[data-testid="stFormSubmitButton"] > button:hover { transform: translateY(-1px); box-shadow: 0 12px 28px rgba(20,184,166,0.38); }
.stDownloadButton > button {
    background: transparent !important; color: var(--text) !important; border: 1px solid var(--line) !important;
}
.stDownloadButton > button:hover { border-color: var(--teal) !important; color: var(--teal) !important; }

/* ---------- Result ---------- */
.verdict {
    background: var(--panel); border: 1px solid var(--line); border-radius: 18px;
    padding: 26px 28px; height: 100%;
}
.verdict-top { display:flex; align-items:center; gap:10px; }
.dot { width: 12px; height: 12px; border-radius: 50%; }
.verdict-label { font-family:'Sora',sans-serif; font-size: 28px; font-weight: 700; letter-spacing:-0.02em; }
.verdict-msg { color: var(--muted); font-size: 15px; margin-top: 8px; line-height: 1.55; }
.gauge-wrap { background: var(--panel); border:1px solid var(--line); border-radius:18px; padding: 18px 18px 10px 18px; }
.gauge { width: 100%; max-width: 330px; display:block; margin: 0 auto; }
.g-num { font-family: 'Sora', sans-serif; font-size: 34px; font-weight: 700; }
.g-cap { font-size: 10.5px; }

/* ---------- Metric cards ---------- */
.mini {
    background: var(--panel); border:1px solid var(--line); border-radius: 14px; padding: 16px 20px; margin-top: 14px;
}
.mini-label { color: var(--muted); font-size: 13.5px; }
.mini-value { font-family:'Sora',sans-serif; font-size: 26px; font-weight: 700; margin-top: 2px; }

/* ---------- Factor rows ---------- */
.factor-card { background: var(--panel); border:1px solid var(--line); border-radius: 18px; padding: 8px 22px; }
.frow { display:flex; align-items:center; justify-content:space-between; padding: 13px 0; border-bottom: 1px solid var(--line); }
.frow:last-child { border-bottom: none; }
.fname { font-size: 15.5px; font-weight: 600; color: #F1F5F9 !important; }
.fval { color: #A9B8CA !important; font-size: 14px; margin-top: 2px; }
.pill { font-size: 12.5px; font-weight: 600; padding: 4px 11px; border-radius: 999px; white-space: nowrap; }
.pill.ok   { color: #2DD4BF; background: rgba(45,212,191,0.12); }
.pill.warn { color: #F5B84B; background: rgba(245,184,75,0.12); }
.pill.bad  { color: #F4727C; background: rgba(244,114,124,0.13); }

/* ---------- Disclaimer + footer ---------- */
.disclaimer {
    margin-top: 22px; padding: 14px 18px; border-radius: 12px; font-size: 13.5px; line-height: 1.55;
    color: #E9C77B; background: rgba(245,184,75,0.07); border: 1px solid rgba(245,184,75,0.22);
}
.footer { text-align:center; color: var(--muted); font-size: 13.5px; padding: 10px 0 6px 0; }
.empty {
    border: 1px dashed var(--line); border-radius: 18px; padding: 34px; text-align:center;
    color: var(--muted); font-size: 15px; margin-top: 18px;
}

/* ---------- Force readable text everywhere (overrides Streamlit's default dark text) ---------- */
.hero-title, .sec-title, .side-title { color: #F1F5F9 !important; }
.hero-sub, .sec-sub, .mini-label, .verdict-msg, .footer, .empty { color: #A9B8CA !important; }
.mini-value { color: #F1F5F9; }  /* no !important so the inline risk colour still wins */
.stApp p, .stApp li, .stApp label { color: #E6EDF3; }

@media (max-width: 768px) { .hero-title { font-size: 32px; } }
</style>
""",
    unsafe_allow_html=True,
)

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    html('<div class="side-title">🩺 Diabetes Risk Assessment</div>')
    html(
        '<p class="side-note">Estimates diabetes risk from routine clinical values such as '
        "glucose, HbA1c and BMI.</p>"
    )
    st.markdown("---")
    html('<div class="side-title" style="font-size:15px;">How it works</div>')
    html(
        """
        <div class="side-step"><b>1</b><span>Enter the patient's details.</span></div>
        <div class="side-step"><b>2</b><span>Select <i>Check risk</i>.</span></div>
        <div class="side-step"><b>3</b><span>Review the risk level and the factors behind it.</span></div>
        """
    )
    st.markdown("---")
    html(
        '<p class="side-note">Model: XGBoost classifier trained on patient health records.<br>'
        "This is a statistical estimate, not a diagnosis.</p>"
    )

# =====================================================
# HEADER
# =====================================================
html(
    """
    <div class="hero">
        <div class="hero-badge">AI-assisted health screening</div>
        <div class="hero-title">Know your diabetes risk<br>in a few seconds.</div>
        <div class="hero-sub">Enter a few routine health values and get an instant risk estimate,
        with a plain-language breakdown of what is driving it.</div>
    </div>
    """
)

# =====================================================
# INPUT FORM
# =====================================================
html('<div class="sec-title">Patient information</div><div class="sec-sub">All fields are required.</div>')

with st.form("patient_form"):
    col1, col2 = st.columns(2, gap="large")

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male", "Other"])
        age = st.slider("Age", 1, 100, 35)
        hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        heart_disease = st.selectbox("Heart disease", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

    with col2:
        bmi = st.number_input("BMI", 10.0, 70.0, 25.5, step=0.1)
        hba1c = st.number_input("HbA1c level (%)", 3.0, 15.0, 5.8, step=0.1)
        glucose = st.number_input("Blood glucose level (mg/dL)", 50, 350, 120)
        smoking = st.selectbox("Smoking history", ["never", "former", "current", "ever", "not current"])

    submitted = st.form_submit_button("Check risk", use_container_width=True)

# =====================================================
# PREDICTION
# =====================================================
if not submitted:
    html('<div class="empty">Your result will appear here after you select <b>Check risk</b>.</div>')
else:
    input_df = pd.DataFrame(
        {
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
            "smoking_history_not current": [1 if smoking == "not current" else 0],
        }
    )

    # Make sure columns match the exact order the model was trained with
    trained_cols = model.get_booster().feature_names
    if trained_cols:
        input_df = input_df.reindex(columns=trained_cols, fill_value=0)

    with st.spinner("Analysing..."):
        proba = model.predict_proba(input_df)[0]

    p_diabetes = float(proba[1])
    p_healthy = float(proba[0])
    label, colour, message = risk_level(p_diabetes)

    st.write("")
    html('<div class="sec-title">Your result</div>')

    left, right = st.columns([1.15, 1], gap="large")

    with left:
        html(
            f"""
            <div class="verdict">
                <div class="verdict-top">
                    <span class="dot" style="background:{colour}; box-shadow:0 0 14px {colour};"></span>
                    <span class="verdict-label" style="color:{colour};">{label}</span>
                </div>
                <div class="verdict-msg">{message}</div>
                <div class="mini">
                    <div class="mini-label">Diabetes probability</div>
                    <div class="mini-value" style="color:{colour};">{p_diabetes*100:.1f}%</div>
                </div>
                <div class="mini">
                    <div class="mini-label">Non-diabetes probability</div>
                    <div class="mini-value">{p_healthy*100:.1f}%</div>
                </div>
            </div>
            """
        )

    with right:
        html(f'<div class="gauge-wrap">{gauge_svg(p_diabetes, colour)}</div>')

    st.write("")
    html(
        '<div class="sec-title">What is driving this result</div>'
        '<div class="sec-sub">Each value compared with common clinical reference ranges.</div>'
    )

    rows = factor_review(age, bmi, hba1c, glucose, hypertension, heart_disease)
    rows_html = "".join(
        f'<div class="frow"><div><div class="fname">{name}</div><div class="fval">{val}</div></div>'
        f'<span class="pill {state}">{note}</span></div>'
        for name, val, note, state in rows
    )
    html(f'<div class="factor-card">{rows_html}</div>')

    # ---- downloadable report ----
    report = (
        "DIABETES RISK ASSESSMENT REPORT\n"
        f"Generated: {datetime.now():%d %b %Y, %H:%M}\n\n"
        f"Result: {label} ({p_diabetes*100:.1f}% diabetes probability)\n\n"
        "Patient details\n"
        f"- Age: {age}\n- Gender: {gender}\n- BMI: {bmi}\n- HbA1c: {hba1c}%\n"
        f"- Blood glucose: {glucose} mg/dL\n"
        f"- Hypertension: {'Yes' if hypertension else 'No'}\n"
        f"- Heart disease: {'Yes' if heart_disease else 'No'}\n"
        f"- Smoking history: {smoking}\n\n"
        "This is a statistical estimate and not a medical diagnosis. "
        "Please consult a qualified healthcare professional.\n"
    )
    st.write("")
    st.download_button(
        "Download report",
        data=report,
        file_name="diabetes_risk_report.txt",
        mime="text/plain",
        use_container_width=True,
    )

    html(
        """
        <div class="disclaimer">
            This result is a statistical estimate from a machine learning model and is <b>not a medical diagnosis</b>.
            Please consult a qualified healthcare professional for diagnosis and treatment.
        </div>
        """
    )

# =====================================================
# FOOTER
# =====================================================
st.write("")
st.markdown("---")
html('<div class="footer">Developed by <b>Iqra Bibi</b> ❤️</div>')
