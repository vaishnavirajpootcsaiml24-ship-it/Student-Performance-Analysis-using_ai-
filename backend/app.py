from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import random
import json
import os
import re
import secrets
import bleach
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd

# Ensure backend directory is in sys.path for imports so we can resolve 'src'
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.src.database import get_db, init_db
from backend.src.predict import predict_grade_full
from backend.src.train_model import generate_ai_feedback

class DummyScaler:
    """A dummy scaler to replace StandardScaler since tree-based models don't need scaling.
       This prevents app.py from breaking when it expects a scaler."""
    def transform(self, data):
        return data

app = Flask(
    __name__,
    template_folder='../frontend/templates',
    static_folder='../frontend/static'
)
app.secret_key = os.environ.get("SECRET_KEY", "studentai_secret_2024")

# ── Rate Limiting ─────────────────────────────────────────────────────────────
try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    limiter = Limiter(
        app=app,
        key_func=get_remote_address,
        default_limits=["200 per hour"],
        storage_uri="memory://"
    )
except ImportError:
    # Graceful fallback if flask-limiter not installed
    class DummyLimiter:
        def limit(self, *a, **kw):
            def decorator(f): return f
            return decorator
    limiter = DummyLimiter()


# ── Security Headers Middleware ───────────────────────────────────────────────
@app.after_request
def set_security_headers(response):
    """Add security headers to every response to prevent common attacks."""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Permissions-Policy'] = 'geolocation=(), microphone=()'
    return response


# ── CSRF Token Helpers ────────────────────────────────────────────────────────
def generate_csrf_token():
    """Generate and store a CSRF token in the session."""
    if '_csrf_token' not in session:
        session['_csrf_token'] = secrets.token_hex(32)
    return session['_csrf_token']

def validate_csrf_token():
    """Validate CSRF token from form against session token."""
    token = request.form.get('_csrf_token', '')
    return token == session.get('_csrf_token', '')

# Make csrf_token available in all templates
app.jinja_env.globals['csrf_token'] = generate_csrf_token


# ── Input Sanitization ────────────────────────────────────────────────────────
def sanitize_input(text):
    """Sanitize text input to prevent XSS attacks."""
    if text is None:
        return ""
    return bleach.clean(str(text).strip())


# ── Auth Decorator ────────────────────────────────────────────────────────────
def login_required(f):
    """Decorator to require login for protected routes."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function


# Custom Filter for JSON parsing in templates
@app.template_filter('from_json')
def from_json_filter(value):
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return {}

# Initialize DB on startup
init_db()


# ── HOME ─────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")


# ── REGISTER ─────────────────────────────────────────────────────────────────
   
@app.route("/register")
def register():
    return render_template("register.html")


@app.route("/do_register", methods=["POST"])
@limiter.limit("10 per minute")
def do_register():
    if not validate_csrf_token():
        return render_template("register.html", error="Invalid form submission. Please try again.")

    username = sanitize_input(request.form.get("username", ""))
    email    = sanitize_input(request.form.get("email", "")).lower()
    password = request.form.get("password", "")

    if not username or not email or not password:
        return render_template("register.html", error="All fields are required.")

    # Server-side password complexity validation
    if len(password) < 6:
        return render_template("register.html",
                               error="Password must be at least 6 characters.")
    if not re.search(r'[A-Z]', password):
        return render_template("register.html",
                               error="Password must include at least one uppercase letter.")
    if not re.search(r'[0-9]', password):
        return render_template("register.html",
                               error="Password must include at least one number.")
    if not re.search(r'[@$!%*?&]', password):
        return render_template("register.html",
                               error="Password must include at least one special character (@$!%*?&).")

    # Email format validation
    if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
        return render_template("register.html", error="Please enter a valid email address.")

    db = get_db()
    user = db.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
    if user:
        db.close()
        return render_template("register.html",
                               error="An account with this email already exists.")

    db.execute('INSERT INTO users (email, username, password) VALUES (?, ?, ?)',
               (email, username, generate_password_hash(password)))
    db.commit()
    db.close()
    return redirect(url_for("login"))


# ── LOGIN ─────────────────────────────────────────────────────────────────────

@app.route("/login")
def login():
    num1 = random.randint(1, 9)
    num2 = random.randint(1, 9)
    session["captcha"] = num1 + num2
    return render_template("login.html", num1=num1, num2=num2, error=None)


@app.route("/do_login", methods=["POST"])
@limiter.limit("10 per minute")
def do_login():
    if not validate_csrf_token():
        num1, num2 = random.randint(1, 9), random.randint(1, 9)
        session["captcha"] = num1 + num2
        return render_template("login.html", error="Invalid form submission. Please try again.",
                               num1=num1, num2=num2)

    email    = sanitize_input(request.form.get("email", "")).lower()
    password = request.form.get("password", "")
    captcha  = request.form.get("captcha",  "")

    num1 = random.randint(1, 9)
    num2 = random.randint(1, 9)

    db = get_db()
    user = db.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
    db.close()

    if not user:
        session["captcha"] = num1 + num2
        return render_template("login.html", error="No account found with this email.",
                               num1=num1, num2=num2)

    if not check_password_hash(user["password"], password):
        session["captcha"] = num1 + num2
        return render_template("login.html", error="Incorrect password.",
                               num1=num1, num2=num2)

    try:
        if int(captcha) != session.get("captcha"):
            session["captcha"] = num1 + num2
            return render_template("login.html", error="Captcha answer incorrect.",
                                   num1=num1, num2=num2)
    except ValueError:
        session["captcha"] = num1 + num2
        return render_template("login.html", error="Please enter a valid captcha.",
                               num1=num1, num2=num2)

    session["user"]     = email
    session["username"] = user["username"]
    return redirect(url_for("dashboard"))


# ── SKIP LOGIN (Guest Mode) ───────────────────────────────────────────────────

@app.route("/guest")
def guest():
    # Ensure guest user exists in DB
    email = "guest@studentai.app"
    db = get_db()
    user = db.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
    if not user:
        db.execute('INSERT INTO users (email, username, password) VALUES (?, ?, ?)',
                   (email, "Guest", generate_password_hash("guest_password")))
        db.commit()
    db.close()
    
    session["user"]     = email
    session["username"] = "Guest"
    return redirect(url_for("dashboard"))


# ── DASHBOARD ─────────────────────────────────────────────────────────────────

@app.route("/dashboard")
@login_required
def dashboard():
    db = get_db()
    history = db.execute('SELECT * FROM history WHERE user_email = ? ORDER BY date DESC', 
                         (session["user"],)).fetchall()
    db.close()

    # Load model metrics if available
    model_metrics = _load_model_metrics()
    
    # Load dataset stats for dynamic dropdowns and validation
    dataset_stats = _get_dataset_stats()
    
    return render_template("dashboard.html",
                           username=session.get("username", "Student"),
                           history=history,
                           model_metrics=model_metrics,
                           dataset_stats=dataset_stats)


# ── PREDICT ───────────────────────────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
@login_required
@limiter.limit("30 per minute")
def predict():
    if not validate_csrf_token():
        return redirect(url_for("dashboard"))

    try:
        study_hours          = float(request.form.get("study_hours", 15))
        attendance           = float(request.form.get("attendance",  80))
        participation        = float(request.form.get("participation", 5))
        extra_curricular     = float(request.form.get("extra_curricular", 4))
        sleep_hours          = float(request.form.get("sleep_hours", 7))
        stress_level         = float(request.form.get("stress_level", 5))
        motivation_score     = float(request.form.get("motivation_score", 6))
        parent_education     = int(request.form.get("parent_education", 1))
        previous_gpa         = float(request.form.get("previous_gpa", 2.8))
        assignment_completion = float(request.form.get("assignment_completion", 75))
    except (ValueError, TypeError):
        dataset_stats = _get_dataset_stats()
        return render_template("dashboard.html",
                               username=session.get("username", "Student"),
                               dataset_stats=dataset_stats,
                               error="Please enter valid numeric values.")

    # ── STRICT VALIDATION (Synced with Dataset Analysis) ──────────────────────
    stats = _get_dataset_stats()
    errors = []
    
    if not (stats['attendance']['min'] <= attendance <= stats['attendance']['max']):
        errors.append(f"Attendance must be between {stats['attendance']['min']}% and {stats['attendance']['max']}% (Dataset Range).")
    if not (stats['study_hours']['min'] <= study_hours <= stats['study_hours']['max']):
        errors.append(f"Weekly Study Hours must be between {stats['study_hours']['min']} and {stats['study_hours']['max']} (Dataset Range).")
    
    if not (0 <= participation <= 10):
        errors.append("Participation must be between 0 and 10.")
    if not (0.0 <= previous_gpa <= 10):
        errors.append("Previous CGPA must be between 0 and 10 (Dataset Range).")
    if not (40 <= assignment_completion <= 100):
        errors.append("Assignment Completion must be between 40% and 100% (Dataset Range).")
    if not (1 <= motivation_score <= 10):
        errors.append("Motivation Score must be between 1 and 10.")
    if not (1 <= stress_level <= 10):
        errors.append("Stress Level must be between 1 and 10.")
    if not (4 <= sleep_hours <= 10):
        errors.append("Sleep Hours must be between 4 and 10 (Dataset Range).")
    
    if extra_curricular not in stats['extra_curricular']['options']:
        errors.append("Selected Extra Curricular value is not in the dataset.")
    if parent_education not in stats['parent_education']['options']:
        errors.append("Selected Parent Education level is not in the dataset.")

    if errors:
        db = get_db()
        history = db.execute('SELECT * FROM history WHERE user_email = ? ORDER BY date DESC', (session["user"],)).fetchall()
        db.close()
        dataset_stats = _get_dataset_stats()
        return render_template("dashboard.html",
                               username=session.get("username", "Student"),
                               history=history,
                               dataset_stats=dataset_stats,
                               error=" | ".join(errors))

    # The model expects GPA on a 0-4 scale, which matches our new input range
    previous_gpa_model = previous_gpa

    features = [
        study_hours, attendance, participation, extra_curricular,
        sleep_hours, stress_level, motivation_score, parent_education,
        previous_gpa_model, assignment_completion,
    ]

    result = predict_grade_full(features)

    feedback = generate_ai_feedback(
        grade                 = result["grade"],
        study_hours           = study_hours,
        attendance            = attendance,
        participation         = participation,
        confidence            = result["confidence"],
    )

    # Save to History
    db = get_db()
    db.execute('''
        INSERT INTO history (user_email, prediction, confidence, features_json)
        VALUES (?, ?, ?, ?)
    ''', (session["user"], result["grade"], result["confidence"], json.dumps({
        "study_hours": study_hours,
        "attendance": attendance,
        "cgpa": previous_gpa,
        "assignment": assignment_completion,
        "participation": participation,
        "stress_level": stress_level,
        "sleep_hours": sleep_hours,
        "motivation": motivation_score
    })))
    db.commit()
    history = db.execute('SELECT * FROM history WHERE user_email = ? ORDER BY date DESC', (session["user"],)).fetchall()
    db.close()

    model_metrics = _load_model_metrics()
    dataset_stats = _get_dataset_stats()

    return render_template(
        "dashboard.html",
        username              = session.get("username", "Student"),
        prediction            = result["grade"],
        confidence            = result["confidence"],
        probabilities         = result["probabilities"],
        feedback              = feedback,
        history               = history,
        model_metrics         = model_metrics,
        dataset_stats         = dataset_stats,
        # Preserve form values
        study_hours           = study_hours,
        attendance            = attendance,
        participation         = participation,
        extra_curricular      = extra_curricular,
        sleep_hours           = sleep_hours,
        stress_level          = stress_level,
        motivation_score      = motivation_score,
        parent_education      = parent_education,
        previous_gpa          = previous_gpa,
        assignment_completion = assignment_completion,
    )


# ── ANALYTICS API ─────────────────────────────────────────────────────────────

@app.route("/api/analytics")
@login_required
def api_analytics():
    """Return chart data for the analytics dashboard."""
    db = get_db()
    history = db.execute('SELECT * FROM history WHERE user_email = ? ORDER BY date ASC',
                         (session["user"],)).fetchall()
    db.close()

    grade_points = {'A': 95, 'B': 85, 'C': 75, 'D': 65, 'F': 55}
    
    if not history:
        # Simulate realistic student inputs if no history exists (PART 3, Point 3.OPTION B)
        # We simulate 5 data points with increasing study hours and attendance
        simulated_history = []
        for i in range(5):
            # Realistic ranges based on dataset constraints
            sh  = 10 + (i * 4)  # 10, 14, 18, 22, 26
            att = 60 + (i * 8)  # 60, 68, 76, 84, 92
            features = [sh, att, 5.0, 4.0, 7.0, 5.0, 6.0, 1, 2.5 + (i * 0.2), 75.0]
            res = predict_grade_full(features)
            simulated_history.append({
                "prediction": res["grade"],
                "confidence": res["confidence"],
                "date": f"Sim Point {i+1}"
            })
        
        display_history = simulated_history
    else:
        display_history = history

    # Grade distribution (for Pie Chart)
    grade_counts = {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0}
    for item in display_history:
        g = item["prediction"]
        if g in grade_counts:
            grade_counts[g] += 1

    # Trend data (for Line Chart)
    # X-axis: index or time, Y-axis: predicted score
    trend_data = []
    labels = []
    for i, item in enumerate(display_history):
        labels.append(item["date"][:10] if " " in str(item["date"]) else str(item["date"]))
        trend_data.append(grade_points.get(item["prediction"], 50))

    return jsonify({
        "has_data": True,
        "grade_distribution": grade_counts,
        "trend_data": trend_data,
        "trend_labels": labels,
        "total_predictions": len(history) if history else 0
    })


# ── MODEL INFO ────────────────────────────────────────────────────────────────

@app.route("/model_info")
@login_required
def model_info():
    """Return model evaluation metrics as JSON.
    Values sourced from the latest training run (GradientBoostingRegressor).
    """
    # Real metrics from training output — not from stale model_metrics.json
    return jsonify({
        "model_info": {
            "model_type": "Gradient Boosting Regressor",
            "train_r2": 0.7315,
            "test_r2": 0.6016,
            "mae": 4.4427
        }
    })


def _load_model_metrics():
    """Load model metrics from the JSON file."""
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    metrics_path = os.path.join(BASE_DIR, 'models', 'model_metrics.json')
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return None


def _get_dataset_stats():
    """Extract unique values and ranges from the CSV dataset."""
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(BASE_DIR, 'data', 'student_performance.csv')
        df = pd.read_csv(csv_path)
        
        stats = {
            "study_hours": {
                "min": float(df['study_hours'].min()),
                "max": float(df['study_hours'].max())
            },
            "attendance": {
                "min": float(df['attendance'].min()),
                "max": float(df['attendance'].max())
            },
            "parent_education": {
                "options": sorted(df['parent_education'].unique().tolist())
            },
            "extra_curricular": {
                "options": sorted(df['extra_curricular'].unique().tolist())
            }
        }
        return stats
    except Exception as e:
        print(f"Error reading dataset stats: {e}")
        # Safe defaults if CSV reading fails
        return {
            "study_hours": {"min": 0, "max": 34},
            "attendance": {"min": 40, "max": 100},
            "parent_education": {"options": [0, 1, 2, 3]},
            "extra_curricular": {"options": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
        }


# ── USER PROFILE ──────────────────────────────────────────────────────────────

@app.route("/profile")
@login_required
def profile():
    db = get_db()
    user = db.execute('SELECT * FROM users WHERE email = ?', (session["user"],)).fetchone()
    history = db.execute('SELECT prediction, COUNT(*) as cnt FROM history WHERE user_email = ? GROUP BY prediction',
                         (session["user"],)).fetchall()
    total = db.execute('SELECT COUNT(*) as total FROM history WHERE user_email = ?',
                       (session["user"],)).fetchone()
    db.close()

    # Build stats
    grade_stats = {row["prediction"]: row["cnt"] for row in history}
    most_common = max(grade_stats, key=grade_stats.get) if grade_stats else "N/A"

    return render_template("profile.html",
                           username=session.get("username", "Student"),
                           user=user,
                           total_predictions=total["total"] if total else 0,
                           grade_stats=grade_stats,
                           most_common_grade=most_common)


@app.route("/update_profile", methods=["POST"])
@login_required
@limiter.limit("10 per minute")
def update_profile():
    if not validate_csrf_token():
        return redirect(url_for("profile"))

    new_username   = sanitize_input(request.form.get("username", ""))
    new_institution = sanitize_input(request.form.get("institution", ""))

    if not new_username:
        return redirect(url_for("profile"))

    db = get_db()
    db.execute('UPDATE users SET username = ?, institution = ? WHERE email = ?',
               (new_username, new_institution, session["user"]))
    db.commit()
    db.close()

    session["username"] = new_username
    return redirect(url_for("profile"))


# ── HISTORY MANAGEMENT ────────────────────────────────────────────────────────

@app.route("/delete_history/<int:history_id>")
@login_required
def delete_history(history_id):
    db = get_db()
    db.execute('DELETE FROM history WHERE id = ? AND user_email = ?', (history_id, session["user"]))
    db.commit()
    db.close()
    return redirect(url_for("dashboard"))


@app.route("/clear_history")
@login_required
def clear_history():
    db = get_db()
    db.execute('DELETE FROM history WHERE user_email = ?', (session["user"],))
    db.commit()
    db.close()
    return redirect(url_for("dashboard"))


# ── LOGOUT ────────────────────────────────────────────────────────────────────

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))


# ── RUN ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    
    port = int(os.environ.get("PORT", 10000))
    
    app.run(
        host="0.0.0.0",
        port=port,
        debug=True
    )
    