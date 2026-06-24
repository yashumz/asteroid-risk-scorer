# predictor.py
# ─────────────────────────────────────────────────────────────────
# The ML brain of the API
# Loads model.pkl once at startup, exposes a predict() function
# that any endpoint can call with raw NEO data
#
# WHY a separate file?
# main.py handles HTTP — it shouldn't care about ML logic
# predictor.py handles ML — it shouldn't care about HTTP
# Clean separation = easier to test, debug, and swap models later
# ─────────────────────────────────────────────────────────────────

import pickle
import numpy as np
import pandas as pd
from pathlib import Path


# ── Load model once at import time ────────────────────────────────
# When FastAPI starts, it imports predictor.py ONCE
# The model loads into memory once and stays there
# Every prediction request reuses the same loaded model
# This is far faster than loading model.pkl on every single request

MODEL_PATH = Path(__file__).parent / "model.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

print(f"Model loaded from {MODEL_PATH}")


# ── Risk tier mapping ──────────────────────────────────────────────
# Convert raw probability (0.0 to 1.0) into a human-readable tier
# These thresholds are our design decision — adjustable based on needs
def get_risk_tier(probability: float) -> str:
    """
    Convert a raw probability score into a risk tier label.

    Args:
        probability: float between 0.0 and 1.0

    Returns:
        Risk tier string: Low / Medium / High / Critical
    """
    if probability < 0.25:
        return "Low"
    elif probability < 0.50:
        return "Medium"
    elif probability < 0.75:
        return "High"
    else:
        return "Critical"


# ── Feature engineering ────────────────────────────────────────────
def engineer_features(neo_dict: dict) -> pd.DataFrame:
    """
    Takes a parsed NEO dict and returns a DataFrame ready for model.predict()
    This is the EXACT same logic as train_model.py — must match perfectly
    If you change features here you must retrain the model too

    Args:
        neo_dict: flat dict with keys: diameter_km, velocity_kps,
                  miss_dist_km, magnitude_h

    Returns:
        Single-row DataFrame with all 10 feature columns
    """
    # Put the raw fields into a one-row DataFrame
    df = pd.DataFrame([neo_dict])

    # Apply identical feature engineering as train_model.py
    df['kinetic_energy_proxy'] = 0.5 * (df['diameter_km'] ** 3) * (df['velocity_kps'] ** 2)
    df['threat_score']         = df['diameter_km'] / (df['miss_dist_km'] / 1e6)
    df['is_fast']              = (df['velocity_kps'] > 20).astype(int)
    df['size_category']        = pd.cut(
        df['diameter_km'],
        bins=[0, 0.3, 1.0, float('inf')],
        labels=[0, 1, 2]
    ).astype(int)
    df['log_miss_dist']        = np.log1p(df['miss_dist_km'])
    df['log_diameter']         = np.log1p(df['diameter_km'])

    # Return ONLY the 10 feature columns in the EXACT order used during training
    # Order matters — if columns are in wrong order, predictions will be wrong
    feature_cols = [
        'diameter_km', 'velocity_kps', 'miss_dist_km', 'magnitude_h',
        'kinetic_energy_proxy', 'threat_score', 'is_fast', 'size_category',
        'log_miss_dist', 'log_diameter'
    ]
    return df[feature_cols]


# ── Main predict function ──────────────────────────────────────────
def predict_risk(neo_dict: dict) -> dict:
    """
    Full prediction pipeline for one asteroid.
    Takes raw parsed NEO data, returns complete risk assessment.

    Args:
        neo_dict: dict with at minimum these keys:
                  id, name, diameter_km, velocity_kps,
                  miss_dist_km, magnitude_h

    Returns:
        dict with risk_score, risk_tier, is_hazardous_predicted,
        top_features, and original neo fields
    """
    # Step 1: Engineer features
    X = engineer_features(neo_dict)

    # Step 2: Get probability of being hazardous
    # predict_proba returns [[prob_safe, prob_hazardous]]
    # [0][1] = first row, second column = hazardous probability
    probability = float(model.predict_proba(X)[0][1])

    # Step 3: Hard prediction (True/False)
    prediction = bool(model.predict(X)[0])

    # Step 4: Get top 3 contributing features using model's feature importance
    importance = model.get_booster().get_score(importance_type='gain')
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]
    top_feature_names = [f[0] for f in top_features]

    # Step 5: Assemble the full response dict
    return {
        "id":                    neo_dict.get('id', 'unknown'),
        "name":                  neo_dict.get('name', 'unknown'),
        "risk_score":            round(probability, 4),
        "risk_percentage":       round(probability * 100, 1),
        "risk_tier":             get_risk_tier(probability),
        "is_hazardous_predicted":prediction,
        "is_hazardous_nasa":     neo_dict.get('is_hazardous', None),
        "top_3_features":        top_feature_names,
        "diameter_km":           neo_dict.get('diameter_km'),
        "velocity_kps":          neo_dict.get('velocity_kps'),
        "miss_dist_km":          neo_dict.get('miss_dist_km'),
        "approach_date":         neo_dict.get('approach_date'),
    }