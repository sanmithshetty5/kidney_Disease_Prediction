from flask import Flask, render_template, request
import pickle
import numpy as np
import os

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

app = Flask(__name__)

# Load models
model = pickle.load(open('model/model.pkl', 'rb'))
scaler = pickle.load(open('model/scaler.pkl', 'rb'))
encoder = pickle.load(open('model/encoder.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Example: assume 3 inputs
        input1 = float(request.form['feature1'])
        input2 = float(request.form['feature2'])
        input3 = float(request.form['feature3'])

        data = np.array([[input1, input2, input3]])

        # Preprocess
        data_scaled = scaler.transform(data)

        # Predict
        prediction = model.predict(data_scaled)

        # Decode (if classification)
        output = encoder.inverse_transform(prediction)[0]

        return render_template('index.html', prediction_text=f'Result: {output}')

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)
