from flask import Flask, jsonify
import tensorflow as tf
import numpy as np
import pandas as pd
import requests
import os
from functools import lru_cache
import time

app = Flask(__name__)

# Load the trained model once at startup
model = tf.keras.models.load_model('sensor_model_tensorflow.keras')

# Cache the API responses for 60 seconds to reduce redundant requests
@lru_cache(maxsize=1)
def fetch_sensor_data():
    try:
        url = 'https://api-ptf3uw5gyq-uc.a.run.app/sensor-data/ml'
        response = requests.get(url, timeout=10)  # Added timeout to avoid hanging
        if response.status_code == 200:
            return response.json().get('data', [])
        return []
    except requests.exceptions.RequestException as e:
        print(f"Error fetching sensor data: {e}")
        return []

# Preprocess for prediction
def preprocess_latest_data(latest_data, df):
    df = pd.DataFrame(df)
    df['gasLevel'] = df['gasLevel'].astype(float)
    df['humidity'] = df['humidity'].astype(float)
    df['temperature'] = df['temperature'].astype(float)

    # Calculate moving averages (avoid calculating if data is too small)
    df['gasLevel_ma'] = df['gasLevel'].rolling(window=3).mean()
    df['temperature_ma'] = df['temperature'].rolling(window=3).mean()
    df['humidity_ma'] = df['humidity'].rolling(window=3).mean()
    df.dropna(inplace=True)

    if len(df) < 1:
        return None

    latest_gasLevel_ma = df['gasLevel_ma'].iloc[-1]
    latest_temperature_ma = df['temperature_ma'].iloc[-1]
    latest_humidity_ma = df['humidity_ma'].iloc[-1]

    timestamp = pd.to_datetime(latest_data['timestamp'], unit='ms')
    hour = timestamp.hour
    day_of_week = timestamp.dayofweek

    X_new = np.array([[latest_data['gasLevel'],
                       latest_data['humidity'],
                       latest_data['temperature'],
                       hour,
                       day_of_week,
                       latest_gasLevel_ma,
                       latest_temperature_ma,
                       latest_humidity_ma]])

    return X_new

@app.route('/predict-latest', methods=['GET'])
def predict_latest():
    # Use cached sensor data to minimize redundant calls
    history_data = fetch_sensor_data()

    # Fetch the latest data for prediction
    latest_data = history_data[-1] if history_data else None

    if not latest_data or not history_data:
        return jsonify({'error': 'Failed to fetch data'}), 500

    # Preprocess data for prediction
    X_new = preprocess_latest_data(latest_data, history_data)

    if X_new is None:
        return jsonify({'error': 'Preprocessing failed'}), 500

    # Reshape input for model prediction
    X_new = np.reshape(X_new, (1, -1))
    prediction = model.predict(X_new)
    label = "Hazardous" if prediction[0][0] >= 0.5 else "Normal"

    return jsonify({
        'prediction': label,
        'confidence': float(prediction[0][0])
    })

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), threaded=True)
