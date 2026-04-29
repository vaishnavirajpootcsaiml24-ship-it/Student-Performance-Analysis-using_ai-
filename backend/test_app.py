"""
test_app.py — Automated pytest tests for the Student Performance Predictor.

Tests cover:
  1. Model loading and prediction
  2. Flask API /predict route (valid input)
  3. Flask API /predict route (invalid input — out-of-range values)
  4. Flask API /model_info route
  5. Guest login flow
"""

import os
import pickle
import pytest
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")


# ═══════════════════════════════════════════════════════════════════════════════
#  1. MODEL TESTS — verify the saved model loads and predicts
# ═══════════════════════════════════════════════════════════════════════════════

class TestModel:
    """Tests that operate directly on the saved model.pkl file."""

    def test_model_file_exists(self):
        """model.pkl must exist in the models/ directory."""
        assert os.path.exists(MODEL_PATH), f"Model file not found at {MODEL_PATH}"

    def test_model_loads_successfully(self):
        """Pickle should load without errors."""
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        assert model is not None

    def test_model_returns_prediction(self):
        """Given a valid 16-feature input, the model should return a prediction."""
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)

        # 16 features matching training order:
        # [sh, att, par, ec, gpa, asc, slp, st, mot, ped,
        #  engagement_score, stress_sleep_ratio, gpa_study_interaction,
        #  participation_efficiency, study_hours_squared, extra_curricular_high]
        sample = np.array([[
            15.0,   # study_hours
            80.0,   # attendance
            5.0,    # participation
            4.0,    # extra_curricular
            2.8,    # previous_gpa
            75.0,   # assignment_completion
            7.0,    # sleep_hours
            5.0,    # stress_level
            6.0,    # motivation_score
            1,      # parent_education
            0.6,    # engagement_score  (att/100 * asc)
            0.625,  # stress_sleep_ratio (st / (slp+1))
            42.0,   # gpa_study_interaction (gpa * sh)
            0.3125, # participation_efficiency (par / (sh+1))
            225.0,  # study_hours_squared
            1       # extra_curricular_high (1 if ec > median)
        ]])

        prediction = model.predict(sample)
        assert prediction is not None
        assert len(prediction) == 1

    def test_predict_proba_returns_confidence(self):
        """predict_proba should return probability array for confidence."""
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)

        sample = np.array([[
            15.0, 80.0, 5.0, 4.0, 2.8, 75.0, 7.0, 5.0, 6.0, 1,
            0.6, 0.625, 42.0, 0.3125, 225.0, 1
        ]])

        # predict_proba should exist and return probabilities
        assert hasattr(model, "predict_proba"), "Model must support predict_proba"
        proba = model.predict_proba(sample)
        assert proba is not None
        assert proba.shape[0] == 1
        # Max probability is the confidence — should be between 0 and 1
        confidence = float(np.max(proba))
        assert 0.0 < confidence <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
#  2. FLASK APP TESTS — verify API routes via the test client
# ═══════════════════════════════════════════════════════════════════════════════

CSRF_TEST_TOKEN = "test_csrf_token_for_pytest"


@pytest.fixture
def client():
    """Create a Flask test client with a guest session and known CSRF token."""
    from app import app

    app.config["TESTING"] = True

    with app.test_client() as client:
        # Log in as guest to establish a session
        client.get("/guest", follow_redirects=True)
        # Set a known CSRF token so POST requests pass validation
        with client.session_transaction() as sess:
            sess["_csrf_token"] = CSRF_TEST_TOKEN
        yield client


class TestFlaskRoutes:
    """Tests that use the Flask test client to hit API endpoints."""

    # ── Valid Prediction ──────────────────────────────────────────────────────

    def test_predict_valid_input(self, client):
        """POST /predict with valid form data should return 200."""
        # First get the dashboard to obtain a CSRF token
        resp = client.get("/dashboard")
        assert resp.status_code == 200

        form_data = {
            "study_hours": "20",
            "attendance": "85",
            "participation": "7",
            "extra_curricular": "3.0",
            "sleep_hours": "7",
            "stress_level": "4",
            "motivation_score": "7",
            "parent_education": "2",
            "previous_gpa": "3.2",
            "assignment_completion": "80",
            "_csrf_token": CSRF_TEST_TOKEN,
        }

        resp = client.post("/predict", data=form_data, follow_redirects=True)
        assert resp.status_code == 200
        # The response should contain the dashboard with a prediction result
        html = resp.data.decode("utf-8")
        assert "Forecasted Outcome" in html

    def test_predict_returns_grade(self, client):
        """The prediction response should contain a grade letter (A–F)."""
        form_data = {
            "study_hours": "25",
            "attendance": "95",
            "participation": "9",
            "extra_curricular": "5.0",
            "sleep_hours": "8",
            "stress_level": "3",
            "motivation_score": "9",
            "parent_education": "3",
            "previous_gpa": "3.8",
            "assignment_completion": "95",
            "_csrf_token": CSRF_TEST_TOKEN,
        }

        resp = client.post("/predict", data=form_data, follow_redirects=True)
        html = resp.data.decode("utf-8")
        # At least one grade letter should appear in the result
        has_grade = any(f"grade-{g}" in html for g in ["A", "B", "C", "D", "F"])
        assert has_grade, "Prediction response must contain a grade"

    # ── Invalid Input ─────────────────────────────────────────────────────────

    def test_predict_invalid_attendance_too_high(self, client):
        """Attendance > 100 should trigger a validation error."""
        form_data = {
            "study_hours": "20",
            "attendance": "150",  # invalid — max is 100
            "participation": "5",
            "extra_curricular": "3.0",
            "sleep_hours": "7",
            "stress_level": "5",
            "motivation_score": "6",
            "parent_education": "1",
            "previous_gpa": "2.8",
            "assignment_completion": "75",
            "_csrf_token": CSRF_TEST_TOKEN,
        }

        resp = client.post("/predict", data=form_data, follow_redirects=True)
        html = resp.data.decode("utf-8")
        assert resp.status_code == 200
        assert "Attendance must be between" in html

    def test_predict_invalid_study_hours_negative(self, client):
        """Negative study hours should trigger a validation error."""
        form_data = {
            "study_hours": "-5",  # invalid — below min
            "attendance": "80",
            "participation": "5",
            "extra_curricular": "3.0",
            "sleep_hours": "7",
            "stress_level": "5",
            "motivation_score": "6",
            "parent_education": "1",
            "previous_gpa": "2.8",
            "assignment_completion": "75",
            "_csrf_token": CSRF_TEST_TOKEN,
        }

        resp = client.post("/predict", data=form_data, follow_redirects=True)
        html = resp.data.decode("utf-8")
        assert resp.status_code == 200
        assert "Weekly Study Hours must be between" in html

    def test_predict_missing_values_uses_defaults(self, client):
        """Missing form fields should fall back to defaults and still respond 200."""
        form_data = {"_csrf_token": CSRF_TEST_TOKEN}  # all fields missing — defaults used

        resp = client.post("/predict", data=form_data, follow_redirects=True)
        assert resp.status_code == 200

    # ── Model Info ────────────────────────────────────────────────────────────

    def test_model_info_returns_json(self, client):
        """GET /model_info should return JSON with model metrics."""
        resp = client.get("/model_info")
        assert resp.status_code == 200

        data = resp.get_json()
        assert "model_info" in data

        info = data["model_info"]
        assert info["model_type"] == "Gradient Boosting Regressor"
        assert info["train_r2"] == 0.7315
        assert info["test_r2"] == 0.6016
        assert info["mae"] == 4.4427

    # ── Guest Login ───────────────────────────────────────────────────────────

    def test_guest_login_redirects_to_dashboard(self, client):
        """GET /guest should redirect to the dashboard."""
        resp = client.get("/guest", follow_redirects=False)
        assert resp.status_code == 302
        assert "/dashboard" in resp.headers["Location"]

    # ── Dashboard Access ──────────────────────────────────────────────────────

    def test_dashboard_accessible_when_logged_in(self, client):
        """GET /dashboard should return 200 when logged in."""
        resp = client.get("/dashboard")
        assert resp.status_code == 200
        html = resp.data.decode("utf-8")
        assert "Academic Parameters" in html

    def test_dashboard_has_dropdowns(self, client):
        """Dashboard form should contain Parent Education and Extracurricular dropdowns."""
        resp = client.get("/dashboard")
        html = resp.data.decode("utf-8")
        assert 'name="parent_education"' in html
        assert 'name="extra_curricular"' in html
        # They should be <select> elements, not hidden inputs
        assert "<select" in html

    # ── Protected Route Without Login ─────────────────────────────────────────

    def test_dashboard_redirects_without_login(self):
        """GET /dashboard without login should redirect to /login."""
        from app import app

        app.config["TESTING"] = True
        with app.test_client() as c:
            resp = c.get("/dashboard", follow_redirects=False)
            assert resp.status_code == 302
            assert "/login" in resp.headers["Location"]


# ═══════════════════════════════════════════════════════════════════════════════
#  3. PREDICT MODULE TESTS — verify predict_grade_full directly
# ═══════════════════════════════════════════════════════════════════════════════

class TestPredictModule:
    """Tests that call predict_grade_full() directly."""

    def test_predict_grade_full_returns_dict(self):
        """predict_grade_full should return a dict with grade, confidence, probabilities."""
        from src.predict import predict_grade_full

        features = [15.0, 80.0, 5.0, 4.0, 7.0, 5.0, 6.0, 1, 2.8, 75.0]
        result = predict_grade_full(features)

        assert isinstance(result, dict)
        assert "grade" in result
        assert "confidence" in result
        assert "probabilities" in result

    def test_predict_grade_is_valid_letter(self):
        """The predicted grade should be one of A, B, C, D, F."""
        from src.predict import predict_grade_full

        features = [20.0, 90.0, 8.0, 3.0, 8.0, 3.0, 8.0, 2, 3.5, 90.0]
        result = predict_grade_full(features)

        assert result["grade"] in ["A", "B", "C", "D", "F"], \
            f"Unexpected grade: {result['grade']}"

    def test_confidence_is_between_0_and_1(self):
        """Confidence should be a float between 0 and 1."""
        from src.predict import predict_grade_full

        features = [10.0, 70.0, 4.0, 2.0, 6.0, 6.0, 5.0, 0, 2.0, 60.0]
        result = predict_grade_full(features)

        assert 0.0 < result["confidence"] <= 1.0
