# 🩺 CKD Risk Predictor — Streamlit Web App

An AI-powered Chronic Kidney Disease risk assessment app built with Streamlit.

---

## 📁 Project Structure

```
kidney_app/
├── app.py                        ← Main Streamlit app
├── kidney_disease_model.pkl      ← Trained ML model  (you provide)
├── kidney_scaler.pkl             ← Fitted scaler     (you provide)
├── requirements.txt              ← Python dependencies
├── .streamlit/
│   └── config.toml               ← Streamlit theme config
└── README.md
```

---

## ⚙️ Local Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your model files
Copy your trained model and scaler into the project folder:
```bash
cp kidney_disease_model.pkl kidney_app/
cp kidney_scaler.pkl         kidney_app/
```

### 3. Run locally
```bash
cd kidney_app
streamlit run app.py
```
Opens at → http://localhost:8501

---

## 🚀 Deploy to Streamlit Community Cloud (Free)

### Step 1 — Push to GitHub
```bash
# In your kidney_app folder:
git init
git add .
git commit -m "Initial commit - CKD predictor app"

# Create a new repo on github.com, then:
git remote add origin https://github.com/YOUR_USERNAME/ckd-predictor.git
git push -u origin main
```

> ⚠️ Make sure `kidney_disease_model.pkl` and `kidney_scaler.pkl` are included in the commit.
> If files are >100MB, use Git LFS: `git lfs track "*.pkl"`

### Step 2 — Deploy on Streamlit Cloud
1. Go to → https://share.streamlit.io
2. Sign in with your GitHub account
3. Click **"New app"**
4. Fill in:
   - **Repository**: `YOUR_USERNAME/ckd-predictor`
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Click **"Deploy!"**
6. Wait ~2 minutes → your app is live at:
   `https://YOUR_USERNAME-ckd-predictor-app-XXXX.streamlit.app`

---

## 🔄 How the Preprocessing Works

The app mirrors exactly what was done in the Jupyter notebook:

| Step | What it does |
|------|-------------|
| Binary encoding | yes/no → 1/0, normal/abnormal → 1/0 |
| Ordinal encoding | activity: high=0, moderate=2, low=1 |
| Feature engineering | eGFR CKD stage, Comorbidity score, Creatinine/Urea ratio, BP/Glucose ratio |
| Scaling | Uses the saved `kidney_scaler.pkl` (same as training) |

---

## 🎯 Output Classes

| Code | Label | Meaning |
|------|-------|---------|
| 0 | No Disease | Healthy kidney function |
| 1 | Low Risk | Early markers, monitor closely |
| 2 | Moderate Risk | Consult nephrologist soon |
| 3 | High Risk | Prompt medical attention needed |
| 4 | Severe Disease | Immediate specialist care required |

---

## ⚠️ Disclaimer

This tool is for educational/research purposes only.
Not a substitute for professional medical advice or diagnosis.
