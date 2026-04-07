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
st.title("🧠 Kidney Disease Prediction")
st.write("Enter all patient details")

# -------------------- FEATURE LIST (24 FEATURES) --------------------
numeric_features = [
    'age','bp','sg','al','su','bgr','bu','sc',
    'sod','pot','hemo','pcv','wc','rc'
]

categorical_features = [
    'rbc','pc','pcc','ba','htn','dm','cad','appet','pe','ane'
]

# sanity check
st.caption(f"Total Features = {len(numeric_features) + len(categorical_features)} (should be 24)")

# -------------------- INPUT UI --------------------
input_data = {}

st.subheader("📊 Numeric Inputs")
col1, col2 = st.columns(2)

for i, feature in enumerate(numeric_features):
    if i % 2 == 0:
        input_data[feature] = col1.number_input(feature.upper(), value=0.0)
    else:
        input_data[feature] = col2.number_input(feature.upper(), value=0.0)

st.subheader("🧬 Categorical Inputs")

for feature in categorical_features:
    options = list(encoders[feature].classes_)
    input_data[feature] = st.selectbox(feature.upper(), options)

# -------------------- PREDICTION --------------------
if st.button("Predict"):
    try:
        # Encode categorical
        for feature in categorical_features:
            encoder = encoders[feature]
            input_data[feature] = encoder.transform([input_data[feature]])[0]

        # Correct feature order (VERY IMPORTANT)
        feature_order = numeric_features + categorical_features

        data = [input_data[f] for f in feature_order]
        data = np.array([data])

        # DEBUG (remove later)
        st.write("Input length:", len(data[0]))
        st.write("Scaler expects:", scaler.n_features_in_)

        # Scale
        data_scaled = scaler.transform(data)

        # Predict
        prediction = model.predict(data_scaled)[0]

        # Output
        if prediction == 1:
            st.error("⚠️ High Risk of Kidney Disease")
        else:
            st.success("✅ Low Risk (No Kidney Disease)")

    except Exception as e:
        st.error(f"Error: {e}")
