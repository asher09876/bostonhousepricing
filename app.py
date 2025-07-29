import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model and scaler
regmodel = pickle.load(open("regmodel.pkl", "rb"))
scalar = pickle.load(open("scaling.pkl", "rb"))

# Home route
@app.route('/')
def home():
    return render_template("home.html")

# API route for prediction (accepts JSON)
@app.route("/predict_api", methods=["POST"])
def predict_api():
    data = request.json['data']
    print("Raw data:", data)

    # Convert to numpy array and reshape
    arr = np.array(list(data.values())).reshape(1, -1)
    print("Reshaped array:", arr)

    # Scale the input
    new_data = scalar.transform(arr)

    # Predict using the loaded model
    output = regmodel.predict(new_data)
    print("Prediction:", output[0])

    return jsonify({'prediction': output[0]})

if __name__ == "__main__":
    app.run(debug=True)
