"""
Train the price model and save it to model/house_price_model.joblib

Usage:
    python train_model.py                  # uses/creates synthetic data
    python train_model.py --data my.csv    # uses your own dataset
"""
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from config import ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET
from data_generator import generate

MODEL_PATH = "model/house_price_model.joblib"
METRICS_PATH = "model/metrics.json"


def build_pipeline(regressor):
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ("num", "passthrough", NUMERIC_FEATURES),
    ])
    return Pipeline([("prep", pre), ("model", regressor)])


def train(data_path=None):
    os.makedirs("model", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    if data_path:
        df = pd.read_csv(data_path)
    elif os.path.exists("data/housing_data.csv"):
        df = pd.read_csv("data/housing_data.csv")
    else:
        df = generate()
        df.to_csv("data/housing_data.csv", index=False)

    df = df.dropna(subset=ALL_FEATURES + [TARGET])
    X, y = df[ALL_FEATURES], np.log1p(df[TARGET])      # log target => stable errors
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    candidates = {
        "RandomForest": RandomForestRegressor(n_estimators=250, n_jobs=-1, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=400, learning_rate=0.06,
                                                      max_depth=4, random_state=42),
    }
    best_name, best_pipe, best_r2, results = None, None, -1, {}
    for name, reg in candidates.items():
        pipe = build_pipeline(reg).fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        r2 = r2_score(y_te, pred)
        results[name] = round(float(r2), 4)
        print(f"{name:18s} R2 = {r2:.4f}")
        if r2 > best_r2:
            best_name, best_pipe, best_r2 = name, pipe, r2

    pred = best_pipe.predict(X_te)
    metrics = {
        "best_model": best_name,
        "r2": round(float(best_r2), 4),
        "mae_inr": round(float(mean_absolute_error(np.expm1(y_te), np.expm1(pred)))),
        "mape_pct": round(float(mean_absolute_percentage_error(np.expm1(y_te), np.expm1(pred)) * 100), 2),
        "log_residual_std": round(float(np.std(y_te - pred)), 4),   # used for price range
        "train_rows": int(len(X_tr)),
        "test_rows": int(len(X_te)),
        "all_models_r2": results,
    }
    joblib.dump(best_pipe, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved model ->", MODEL_PATH)
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="Path to your own CSV dataset")
    train(ap.parse_args().data)
