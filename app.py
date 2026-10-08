"""Flask backend: serves the website and the /api/predict endpoint."""
import json
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

import config
from train_model import METRICS_PATH, MODEL_PATH, train

app = Flask(__name__)

# Train automatically on first run so a fresh GitHub clone "just works"
if not os.path.exists(MODEL_PATH):
    print("No model found - training one now (takes ~1 minute)...")
    train()

model = joblib.load(MODEL_PATH)
with open(METRICS_PATH) as f:
    metrics = json.load(f)


def validate(payload: dict):
    """Returns (clean_row, error_message)."""
    row = {}
    for col in config.NUMERIC_FEATURES:
        if col not in payload or payload[col] in ("", None):
            return None, f"Missing value: {col.replace('_', ' ')}"
        try:
            val = float(payload[col])
        except (TypeError, ValueError):
            return None, f"{col.replace('_', ' ')} must be a number"
        lo, hi = config.NUMERIC_LIMITS[col]
        if not lo <= val <= hi:
            return None, f"{col.replace('_', ' ')} must be between {lo} and {hi}"
        row[col] = val

    choices = {
        "city": config.CITIES, "property_type": config.PROPERTY_TYPES,
        "parking": config.PARKING, "furnishing": config.FURNISHING,
        "facing": config.FACING,
    }
    for col, allowed in choices.items():
        if payload.get(col) not in allowed:
            return None, f"Choose a valid option for {col.replace('_', ' ')}"
        row[col] = payload[col]
    return row, None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/options")
def options():
    return jsonify({
        "cities": config.CITIES, "property_types": config.PROPERTY_TYPES,
        "parking": config.PARKING, "furnishing": config.FURNISHING,
        "facing": config.FACING, "limits": config.NUMERIC_LIMITS,
    })


@app.route("/api/metrics")
def get_metrics():
    return jsonify(metrics)


@app.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Send the property details as JSON."}), 400
    row, err = validate(payload)
    if err:
        return jsonify({"error": err}), 422

    X = pd.DataFrame([row])[config.ALL_FEATURES]
    log_pred = float(model.predict(X)[0])
    price = float(np.expm1(log_pred))
    spread = 1.28 * metrics["log_residual_std"]          # ~80% interval
    low, high = float(np.expm1(log_pred - spread)), float(np.expm1(log_pred + spread))

    return jsonify({
        "price": round(price, -3),
        "low": round(low, -3),
        "high": round(high, -3),
        "price_per_sqft": round(price / row["area_sqft"]),
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
