import streamlit as st
import numpy as np
import joblib
import pandas as pd

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CKD Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    background: #0d1117;
    color: #e6edf3;
}

/* ── Header ── */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
}
.hero h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 3rem;
    color: #e6edf3;
    letter-spacing: -1px;
    margin-bottom: 0.3rem;
}
.hero h1 span { color: #3fb950; }
.hero p {
    color: #8b949e;
    font-size: 1.05rem;
    font-weight: 300;
    margin-top: 0;
}

/* ── Section headers ── */
.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #3fb950;
    margin: 2rem 0 0.8rem;
    border-left: 3px solid #3fb950;
    padding-left: 0.6rem;
}

/* ── Inputs ── */
div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] > div > div {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    color: #e6edf3 !important;
}
div[data-testid="stNumberInput"] input:focus,
div[data-testid="stSelectbox"] > div > div:focus {
    border-color: #3fb950 !important;
    box-shadow: 0 0 0 2px rgba(63,185,80,0.2) !important;
}
label { color: #8b949e !important; font-size: 0.82rem !important; }

/* ── Predict button ── */
div[data-testid="stFormSubmitButton"] button {
    width: 100%;
    background: #238636 !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    padding: 0.9rem !important;
    margin-top: 1.5rem;
    letter-spacing: 0.5px;
    transition: background 0.2s ease !important;
}
div[data-testid="stFormSubmitButton"] button:hover {
    background: #2ea043 !important;
}

/* ── Result card ── */
.result-card {
    border-radius: 14px;
    padding: 2rem 2.5rem;
    margin: 2rem 0 1rem;
    border: 1px solid #30363d;
    animation: fadeUp 0.5s ease;
}
@keyframes fadeUp {
    from { opacity:0; transform:translateY(16px); }
    to   { opacity:1; transform:translateY(0); }
}
.result-stage {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    margin-bottom: 0.3rem;
}
.result-sub {
    color: #8b949e;
    font-size: 0.95rem;
    margin-bottom: 1.2rem;
}
.result-advice {
    font-size: 0.92rem;
    line-height: 1.7;
    color: #c9d1d9;
}

/* Risk-level colours */
.risk-0 { background:#0d2818; border-color:#238636; }
.risk-0 .result-stage { color:#3fb950; }
.risk-1 { background:#1c2a14; border-color:#56d364; }
.risk-1 .result-stage { color:#56d364; }
.risk-2 { background:#272212; border-color:#e3b341; }
.risk-2 .result-stage { color:#e3b341; }
.risk-3 { background:#2d1a12; border-color:#f85149; }
.risk-3 .result-stage { color:#f85149; }
.risk-4 { background:#1f0c0c; border-color:#da3633; }
.risk-4 .result-stage { color:#da3633; }

/* ── Probability bar ── */
.prob-row { display:flex; align-items:center; gap:10px; margin:6px 0; }
.prob-label { width:160px; font-size:0.8rem; color:#8b949e; flex-shrink:0; }
.prob-bar-bg { flex:1; background:#21262d; border-radius:4px; height:10px; }
.prob-bar-fill { height:10px; border-radius:4px; transition:width 0.6s ease; }
.prob-pct { width:42px; font-size:0.8rem; color:#c9d1d9; text-align:right; }

/* ── Divider ── */
hr { border-color:#21262d !important; }

/* ── Disclaimer ── */
.disclaimer {
    background:#161b22;
    border:1px solid #30363d;
    border-radius:8px;
    padding:1rem 1.2rem;
    font-size:0.78rem;
    color:#6e7681;
    margin-top:1.5rem;
    line-height:1.6;
}
</style>
""", unsafe_allow_html=True)

# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model  = joblib.load("kidney_disease_model.pkl")
    scaler = joblib.load("kidney_scaler.pkl")
    return model, scaler

try:
    model, scaler = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ── Constants ──────────────────────────────────────────────────────────────────
CLASSES = ["No Disease", "Low Risk", "Moderate Risk", "High Risk", "Severe Disease"]
CLASS_KEYS = ["No_Disease", "Low_Risk", "Moderate_Risk", "High_Risk", "Severe_Disease"]

ADVICE = {
    0: "✅ Your indicators suggest healthy kidney function. Maintain a balanced diet, stay hydrated, and have annual check-ups.",
    1: "⚠️ Early risk markers detected. Reduce salt and protein intake, monitor blood pressure, and consult your doctor within 3 months.",
    2: "🔶 Moderate risk detected. Please consult a nephrologist soon. Follow a kidney-friendly diet and manage diabetes/hypertension closely.",
    3: "🔴 High risk detected. Seek medical attention promptly. Specialist evaluation and close monitoring are strongly recommended.",
    4: "🚨 Severe risk indicators present. Please visit a nephrologist or emergency care immediately for further evaluation.",
}

BAR_COLORS = ["#3fb950", "#56d364", "#e3b341", "#f85149", "#da3633"]

def egfr_stage(egfr):
    if egfr >= 90: return 0
    elif egfr >= 60: return 1
    elif egfr >= 45: return 2
    elif egfr >= 30: return 3
    elif egfr >= 15: return 4
    else: return 5

def preprocess(inp):
    binary_map = {"yes":1,"no":0,"normal":1,"abnormal":0,"present":1,"not present":0,"good":1,"poor":0}
    activity_map = {"high":0,"low":1,"moderate":2}
    sediment_map = {"abnormal":0,"normal":1}
    smoke_map    = {"no":0,"yes":1}

    row = [
        inp["age"], inp["bp"], inp["sg"], inp["albumin"], inp["sugar"],
        binary_map[inp["rbc"]], binary_map[inp["pc"]], binary_map[inp["pcc"]],
        binary_map[inp["ba"]], inp["bgr"], inp["bu"], inp["sc"], inp["sod"],
        inp["pot"], inp["hemo"], inp["pcv"], inp["wbc"], inp["rbcc"],
        binary_map[inp["htn"]], binary_map[inp["dm"]], binary_map[inp["cad"]],
        binary_map[inp["appet"]], binary_map[inp["pe"]], binary_map[inp["ane"]],
        inp["egfr"], inp["upcr"], inp["uo"], inp["sal"], inp["chol"],
        inp["pth"], inp["ca"], inp["phos"], binary_map[inp["fh"]],
        smoke_map[inp["smoke"]], inp["bmi"],
        activity_map[inp["activity"]], inp["dm_dur"], inp["htn_dur"],
        inp["cystatin"], sediment_map[inp["sediment"]], inp["crp"], inp["il6"],
        # Engineered features
        egfr_stage(inp["egfr"]),
        binary_map[inp["htn"]] + binary_map[inp["dm"]] + binary_map[inp["cad"]] +
            binary_map[inp["ane"]] + binary_map[inp["pe"]],
        inp["sc"] / (inp["bu"] + 1e-5),
        inp["bp"] / (inp["bgr"] + 1e-5),
    ]
    return np.array(row).reshape(1, -1)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🩺 CKD <span>Risk</span> Predictor</h1>
  <p>Enter patient lab values below for an AI-powered Chronic Kidney Disease risk assessment</p>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error("⚠️ Model files not found. Place `kidney_disease_model.pkl` and `kidney_scaler.pkl` in the same folder as `app.py`.")
    st.stop()

st.markdown("<hr>", unsafe_allow_html=True)

# ── Form ───────────────────────────────────────────────────────────────────────
with st.form("prediction_form"):

    # ── Section 1: Patient Demographics ──
    st.markdown('<p class="section-label">Patient Demographics</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    age  = c1.number_input("Age (years)",           min_value=5,   max_value=90,  value=45)
    bmi  = c2.number_input("BMI",                   min_value=15.0, max_value=40.0, value=25.0, step=0.1)
    bp   = c3.number_input("Blood Pressure (mm/Hg)", min_value=80,  max_value=180, value=120)

    # ── Section 2: Urine Analysis ──
    st.markdown('<p class="section-label">Urine Analysis</p>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    sg      = c1.number_input("Specific Gravity",    min_value=1.005, max_value=1.025, value=1.015, step=0.001, format="%.3f")
    albumin = c2.number_input("Albumin (0–5)",        min_value=0, max_value=5, value=0)
    sugar   = c3.number_input("Sugar (0–5)",          min_value=0, max_value=5, value=0)
    upcr    = c4.number_input("Urine Protein:Creatinine", min_value=0.1, max_value=4.5, value=0.5, step=0.1)

    c1, c2, c3, c4 = st.columns(4)
    uo       = c1.number_input("Urine Output (ml/day)", min_value=300, max_value=3000, value=1500)
    rbc      = c2.selectbox("RBC in Urine",      ["normal", "abnormal"])
    pc       = c3.selectbox("Pus Cells",          ["normal", "abnormal"])
    pcc      = c4.selectbox("Pus Cell Clumps",    ["not present", "present"])
    c1, c2 = st.columns(2)
    ba       = c1.selectbox("Bacteria in Urine",  ["not present", "present"])
    sediment = c2.selectbox("Sediment Microscopy",["normal", "abnormal"])

    # ── Section 3: Blood Tests ──
    st.markdown('<p class="section-label">Blood Tests</p>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    bgr  = c1.number_input("Blood Glucose (mg/dl)",    min_value=70,  max_value=500,  value=120)
    bu   = c2.number_input("Blood Urea (mg/dl)",        min_value=7.0, max_value=200.0, value=20.0, step=0.5)
    sc   = c3.number_input("Serum Creatinine (mg/dl)",  min_value=0.5, max_value=15.0, value=1.0, step=0.1)
    sod  = c4.number_input("Sodium (mEq/L)",            min_value=120.0, max_value=150.0, value=135.0, step=0.1)

    c1, c2, c3, c4 = st.columns(4)
    pot   = c1.number_input("Potassium (mEq/L)",        min_value=3.5, max_value=6.5, value=4.5, step=0.1)
    hemo  = c2.number_input("Hemoglobin (gms)",          min_value=6.0, max_value=18.0, value=13.0, step=0.1)
    pcv   = c3.number_input("Packed Cell Volume (%)",   min_value=20,  max_value=55,  value=40)
    wbc   = c4.number_input("WBC Count (cells/cumm)",   min_value=3000, max_value=15000, value=8000)

    c1, c2, c3, c4 = st.columns(4)
    rbcc  = c1.number_input("RBC Count (millions/cumm)", min_value=2.5, max_value=6.0, value=4.5, step=0.1)
    sal   = c2.number_input("Serum Albumin",             min_value=2.0, max_value=4.5, value=3.5, step=0.1)
    chol  = c3.number_input("Cholesterol (mg/dl)",       min_value=100, max_value=300, value=180)
    egfr  = c4.number_input("eGFR",                      min_value=5.0, max_value=120.0, value=90.0, step=0.5)

    c1, c2, c3, c4 = st.columns(4)
    pth      = c1.number_input("PTH Level",              min_value=10.0, max_value=70.0, value=30.0, step=0.5)
    ca       = c2.number_input("Serum Calcium",          min_value=7.5, max_value=10.5, value=9.0, step=0.1)
    phos     = c3.number_input("Serum Phosphate",        min_value=2.5, max_value=6.0, value=3.5, step=0.1)
    cystatin = c4.number_input("Cystatin C Level",       min_value=0.5, max_value=3.0, value=1.0, step=0.1)

    c1, c2 = st.columns(2)
    crp = c1.number_input("CRP Level",                   min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    il6 = c2.number_input("IL-6 Level",                  min_value=0.5, max_value=15.0, value=2.0, step=0.1)

    # ── Section 4: Medical History ──
    st.markdown('<p class="section-label">Medical History & Lifestyle</p>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    htn    = c1.selectbox("Hypertension",        ["no", "yes"])
    dm     = c2.selectbox("Diabetes Mellitus",   ["no", "yes"])
    cad    = c3.selectbox("Coronary Artery Disease", ["no", "yes"])
    ane    = c4.selectbox("Anemia",              ["no", "yes"])

    c1, c2, c3, c4 = st.columns(4)
    pe       = c1.selectbox("Pedal Edema",       ["no", "yes"])
    appet    = c2.selectbox("Appetite",           ["good", "poor"])
    fh       = c3.selectbox("Family History CKD",["no", "yes"])
    smoke    = c4.selectbox("Smoking",            ["no", "yes"])

    c1, c2, c3 = st.columns(3)
    activity = c1.selectbox("Physical Activity", ["high", "moderate", "low"])
    dm_dur   = c2.number_input("Diabetes Duration (yrs)", min_value=0, max_value=30, value=0)
    htn_dur  = c3.number_input("Hypertension Duration (yrs)", min_value=0, max_value=30, value=0)

    submitted = st.form_submit_button("🔬 Predict Kidney Disease Risk")

# ── Prediction ─────────────────────────────────────────────────────────────────
if submitted:
    inputs = dict(
        age=age, bp=bp, sg=sg, albumin=albumin, sugar=sugar,
        rbc=rbc, pc=pc, pcc=pcc, ba=ba, bgr=bgr, bu=bu, sc=sc,
        sod=sod, pot=pot, hemo=hemo, pcv=pcv, wbc=wbc, rbcc=rbcc,
        htn=htn, dm=dm, cad=cad, appet=appet, pe=pe, ane=ane,
        egfr=egfr, upcr=upcr, uo=uo, sal=sal, chol=chol, pth=pth,
        ca=ca, phos=phos, fh=fh, smoke=smoke, bmi=bmi,
        activity=activity, dm_dur=dm_dur, htn_dur=htn_dur,
        cystatin=cystatin, sediment=sediment, crp=crp, il6=il6,
    )

    X = preprocess(inputs)
    X_scaled = scaler.transform(X)
    pred_class = int(model.predict(X_scaled)[0])
    pred_label = CLASSES[pred_class]

    # Probability bars
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_scaled)[0]
    else:
        probs = np.zeros(5)
        probs[pred_class] = 1.0

    # ── Result card ──
    st.markdown(f"""
    <div class="result-card risk-{pred_class}">
      <div class="result-stage">{pred_label}</div>
      <div class="result-sub">AI-predicted CKD risk stage based on provided lab values</div>
      <div class="result-advice">{ADVICE[pred_class]}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Probability bars ──
    st.markdown("**Probability across all risk stages:**")
    bars_html = ""
    for i, (cls, prob, color) in enumerate(zip(CLASSES, probs, BAR_COLORS)):
        pct = prob * 100
        width = max(pct, 1)
        bold = "font-weight:600; color:#e6edf3;" if i == pred_class else ""
        bars_html += f"""
        <div class="prob-row">
          <div class="prob-label" style="{bold}">{cls}</div>
          <div class="prob-bar-bg">
            <div class="prob-bar-fill" style="width:{width}%; background:{color};"></div>
          </div>
          <div class="prob-pct" style="{bold}">{pct:.1f}%</div>
        </div>"""
    st.markdown(bars_html, unsafe_allow_html=True)

    # ── Disclaimer ──
    st.markdown("""
    <div class="disclaimer">
      ⚠️ <strong>Medical Disclaimer:</strong> This tool is for educational and research purposes only.
      It is not a substitute for professional medical advice, diagnosis, or treatment.
      Always consult a qualified healthcare provider for medical decisions.
    </div>
    """, unsafe_allow_html=True)
