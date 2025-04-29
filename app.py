# app.py
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import pickle

# Load the models
m1_model = tf.keras.models.load_model("M1_model.h5")
m9_model = tf.keras.models.load_model("M9_model.h5")

# 🛠 Add this line to load the pickle model
with open("TrafficPrediction.pkl", "rb") as f:
    traffic_model = pickle.load(f)

app = Flask(__name__)

@app.route('/predict/bundle/m1', methods=['POST'])
def predict_bundle_m1():
    data = request.get_json(force=True)
    input_data = np.array(data['input']).reshape(1, -1)
    prediction = m1_model.predict(input_data)
    return jsonify({"prediction": prediction.tolist()})

@app.route('/predict/bundle/m9', methods=['POST'])
def predict_bundle_m9():
    data = request.get_json(force=True)
    input_data = np.array(data['input']).reshape(1, -1)
    prediction = m9_model.predict(input_data)
    return jsonify({"prediction": prediction.tolist()})

@app.route('/predict/bundle/traffic', methods=['POST'])
def predict_bundle_traffic():
    data = request.get_json(force=True)
    input_data = np.array(data['input']).reshape(1, -1)
    prediction = traffic_model.predict(input_data)
    return jsonify({"prediction": prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
