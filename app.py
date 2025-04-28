{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "419a84da-2bc1-429a-a7a7-20812697d139",
   "metadata": {},
   "outputs": [],
   "source": [
    "# app.py\n",
    "from flask import Flask, request, jsonify\n",
    "import tensorflow as tf\n",
    "import numpy as np\n",
    "import pickle\n",
    "\n",
    "# Load the models\n",
    "m1_model = tf.keras.models.load_model(\"M1_model.h5\")\n",
    "m9_model = tf.keras.models.load_model(\"M9_model.h5\")\n",
    "\n",
    "app = Flask(__name__)\n",
    "\n",
    "@app.route('/predict/bundle/m1', methods=['POST'])\n",
    "def predict_bundle_m1():\n",
    "    data = request.get_json(force=True)\n",
    "    input_data = np.array(data['input']).reshape(1, -1)\n",
    "    prediction = m1_model.predict(input_data)\n",
    "    return jsonify({\"prediction\": prediction.tolist()})\n",
    "\n",
    "@app.route('/predict/bundle/m9', methods=['POST'])\n",
    "def predict_bundle_m9():\n",
    "    data = request.get_json(force=True)\n",
    "    input_data = np.array(data['input']).reshape(1, -1)\n",
    "    prediction = m9_model.predict(input_data)\n",
    "    return jsonify({\"prediction\": prediction.tolist()})\n",
    "\n",
    "@app.route('/predict/bundle/traffic', methods=['POST'])\n",
    "def predict_bundle_traffic():\n",
    "    data = request.get_json(force=True)\n",
    "    input_data = np.array(data['input']).reshape(1, -1)\n",
    "    prediction = traffic_model.predict(input_data)\n",
    "    return jsonify({\"prediction\": prediction.tolist()})\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    app.run(host='0.0.0.0', port=5000)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "01d46088-a804-4c46-b44a-6ad979fdc0c1",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.18"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
