import pickle
import numpy as np
import os
from src.train_model import DummyScaler

# Paths to saved models (relative to project root)
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH   = os.path.join(BASE_DIR, "models", "model.pkl")
SCALER_PATH  = os.path.join(BASE_DIR, "models", "scaler.pkl")

_model   = None
_scaler  = None

# Training set median for extra_curricular
EXTRA_CURRICULAR_MEDIAN = 2.1

def _load():
    """Lazy-load the model and scaler using pickle."""
    global _model, _scaler
    if _model is None:
        try:
            if os.path.exists(MODEL_PATH):
                with open(MODEL_PATH, 'rb') as f:
                    _model = pickle.load(f)
            if os.path.exists(SCALER_PATH):
                with open(SCALER_PATH, 'rb') as f:
                    _scaler = pickle.load(f)
        except Exception as e:
            print(f"Error loading model files: {e}")

def predict_grade_full(features):
    """
    Takes 10 input features from the UI, engineers 6 more,
    and returns the predicted grade and dynamic confidence using predict_proba().
    """
    _load()
    
    if _model is None:
        return {
            "grade": "N/A",
            "confidence": 0.0,
            "probabilities": {},
            "error": "Model not loaded"
        }

    # Features from app.py: [sh, att, par, ec, slp, st, mot, ped, gpa, asc]
    try:
        sh  = float(features[0])
        att = float(features[1])
        par = float(features[2])
        ec  = float(features[3])
        slp = float(features[4])
        st  = float(features[5])
        mot = float(features[6])
        ped = int(features[7])
        gpa = float(features[8])
        asc = float(features[9])
    except (IndexError, ValueError):
        # Fallback to defaults
        sh, att, par, ec, slp, st, mot, ped, gpa, asc = 15.0, 80.0, 5.0, 4.0, 7.0, 5.0, 6.0, 1, 2.8, 75.0

    # Feature Engineering (Must match train_model.py)
    engagement_score = (att / 100.0) * asc
    stress_sleep_ratio = st / (slp + 1.0)
    gpa_study_interaction = gpa * sh
    participation_efficiency = par / (sh + 1.0)
    study_hours_squared = sh ** 2
    extra_curricular_high = 1 if ec > EXTRA_CURRICULAR_MEDIAN else 0

    # Construct the 16-feature vector in the correct order
    X_raw = np.array([[
        sh, att, par, ec, gpa, asc, slp, st, mot, ped,
        engagement_score, stress_sleep_ratio, gpa_study_interaction,
        participation_efficiency, study_hours_squared, extra_curricular_high
    ]])
    
    # Scale features
    X_scaled = _scaler.transform(X_raw) if _scaler else X_raw

    # Predict Class and Probabilities
    prediction = _model.predict(X_scaled)[0]
    
    # Map numeric prediction to grade
    if prediction >= 90:
        grade = 'A'
    elif prediction >= 80:
        grade = 'B'
    elif prediction >= 70:
        grade = 'C'
    elif prediction >= 60:
        grade = 'D'
    else:
        grade = 'F'
    
    # Fixed confidence for regressor
    confidence = 0.8
    
    # Empty probabilities dict
    prob_dict = {}

    return {
        "grade": grade,
        "confidence": confidence,
        "probabilities": prob_dict
    }

def predict_grade(features):
    """Simplified prediction returning only the grade string."""
    result = predict_grade_full(features)
    return result["grade"]