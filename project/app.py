import streamlit as st
import pickle
import numpy as np
import os

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Kidney Disease Predictor",
    page_icon="🧠",
    layout="centered"
)

# -------------------- LOAD FILES --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, 'models', 'ckd_rf_model.pkl'), 'rb'))
scaler = pickle.load(open(os.path.join(BASE_DIR, 'models', 'ckd_scaler.pkl'), 'rb'))
encoders = pickle.load(open(os.path.join(BASE_DIR, 'models', 'ckd_label_encoders.pkl'), 'rb'))

# -------------------- UI STYLE --------------------
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #0f172a, #1e293b);
}
.stButton>button {
    background-color: #22c55e;
    color: black;
    font-weight: bold;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
.stButton>button:hover {
    background-color: #16a34a;
}
</style>
""", unsafe_allow_html=True)

# -------------------- TITLE --------------------
st.title("🧠 Kidney Disease Prediction System")
st.write("Fill in patient medical details below")

# -------------------- FEATURE MAP (FULL NAMES) --------------------
feature_labels = {
    'age': "Age",
    'bp': "Blood Pressure (mm Hg)",
    'sg': "Specific Gravity",
    'al': "Albumin",
    'su': "Sugar",
    'rbc': "Red Blood Cells",
    'pc': "Pus Cells",
    'pcc': "Pus Cell Clumps",
    'ba': "Bacteria",
    'bgr': "Blood Glucose Random",
    'bu': "Blood Urea",
    'sc': "Serum Creatinine",
    'sod': "Sodium",
    'pot': "Potassium",
    'hemo': "Hemoglobin",
    'pcv': "Packed Cell Volume",
    'wc': "White Blood Cell Count",
    'rc': "Red Blood Cell Count",
    'htn': "Hypertension",
    'dm': "Diabetes Mellitus",
    'cad': "Coronary Artery Disease",
    'appet': "Appetite",
    'pe': "Pedal Edema",
    'ane': "Anemia"
}

numeric_features = [
    'age','bp','sg','al','su','bgr','bu','sc',
    'sod','pot','hemo','pcv','wc','rc'
]

categorical_features = [
    'rbc','pc','pcc','ba','htn','dm','cad','appet','pe','ane'
]

# -------------------- INPUT UI --------------------
input_data = {}

# 🔹 Numeric Section
st.subheader("📊 Vital & Blood Test Parameters")
col1, col2 = st.columns(2)

for i, feature in enumerate(numeric_features):
    label = feature_labels[feature]
    if i % 2 == 0:
        input_data[feature] = col1.number_input(label, value=0.0)
    else:
        input_data[feature] = col2.number_input(label, value=0.0)

# 🔹 Categorical Section
st.subheader("🧬 Medical Conditions & Urine Analysis")

for feature in categorical_features:
    label = feature_labels[feature]
    options = list(encoders[feature].classes_)
    input_data[feature] = st.selectbox(label, options)

# -------------------- PREDICT --------------------
if st.button("🔍 Predict Kidney Disease"):
    try:
        # Encode categorical
        for feature in categorical_features:
            encoder = encoders[feature]
            input_data[feature] = encoder.transform([input_data[feature]])[0]

        # Maintain EXACT order
        feature_order = numeric_features + categorical_features

        data = [input_data[f] for f in feature_order]
        data = np.array([data])

        # Scale
        data_scaled = scaler.transform(data)

        # Predict class
        prediction = model.predict(data_scaled)[0]

        # Predict probability
        prob = model.predict_proba(data_scaled)[0][1]  # probability of CKD

        st.markdown("---")

        # Show result
        if prediction == 1:
            st.error(f"⚠️ High Risk of Kidney Disease ({prob*100:.2f}% confidence)")
        else:
            st.success(f"✅ Low Risk ({(1-prob)*100:.2f}% confidence)")

        # Progress bar (nice UI touch)
        st.progress(int(prob * 100))

    except Exception as e:
        st.error(f"Error: {e}")
