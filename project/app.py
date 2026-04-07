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

# -------------------- LOAD FILES SAFELY --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, 'models', 'ckd_rf_model.pkl')
scaler_path = os.path.join(BASE_DIR, 'models', 'ckd_scaler.pkl')
encoder_path = os.path.join(BASE_DIR, 'models', 'ckd_label_encoders.pkl')

model = pickle.load(open(model_path, 'rb'))
scaler = pickle.load(open(scaler_path, 'rb'))
label_encoders = pickle.load(open(encoder_path, 'rb'))

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
st.title("🧠 Kidney Disease Prediction App")
st.write("Fill all patient details below")

# -------------------- FEATURE INPUTS --------------------

# You MUST match dataset columns exactly
# Example CKD dataset features (edit if yours differ)

numeric_features = [
    'age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc'
]

categorical_features = [
    'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane'
]

input_data = {}

st.subheader("📊 Patient Information")

# Numeric inputs
for feature in numeric_features:
    input_data[feature] = st.number_input(f"{feature.upper()}", value=0.0)

# Categorical inputs
for feature in categorical_features:
    encoder = label_encoders[feature]
    options = list(encoder.classes_)
    input_data[feature] = st.selectbox(f"{feature.upper()}", options)

# -------------------- PREDICT BUTTON --------------------
if st.button("Predict"):
    try:
        # Convert categorical → encoded
        for feature in categorical_features:
            encoder = label_encoders[feature]
            input_data[feature] = encoder.transform([input_data[feature]])[0]

        # Arrange in correct order
        feature_order = numeric_features + categorical_features
        data = [input_data[f] for f in feature_order]

        data = np.array([data])

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
