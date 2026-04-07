import streamlit as st
import pickle
import numpy as np

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="ML Predictor",
    page_icon="🔍",
    layout="centered"
)

# -------------------- LOAD MODELS --------------------
model = pickle.load(open('models/ckd_rf_model.pkl', 'rb'))
scaler = pickle.load(open('models/ckd_scaler.pkl', 'rb'))

# If you have encoder, uncomment:
encoder = pickle.load(open('models/ckd_label_encoders.pkl', 'rb'))

# -------------------- CUSTOM CSS --------------------
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #0f172a, #1e293b);
    }
    .stButton>button {
        background-color: #22c55e;
        color: black;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px;
    }
    .stButton>button:hover {
        background-color: #16a34a;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------- TITLE --------------------
st.title("🔍 ML Prediction App")
st.write("Enter values below to get prediction")

# -------------------- INPUT FORM --------------------
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        f1 = st.number_input("Feature 1")
        f2 = st.number_input("Feature 2")

    with col2:
        f3 = st.number_input("Feature 3")

    submit = st.form_submit_button("Predict")

# -------------------- PREDICTION --------------------
if submit:
    try:
        data = np.array([[f1, f2, f3]])
        data = scaler.transform(data)

        prediction = model.predict(data)

        # If classification with encoder:
        # prediction = encoder.inverse_transform(prediction)

        st.success(f"Prediction: {prediction[0]}")

    except Exception as e:
        st.error(f"Error: {e}")
