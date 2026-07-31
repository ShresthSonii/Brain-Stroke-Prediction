from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import joblib
from datetime import datetime
import requests
from math import radians, sin, cos, sqrt, atan2

app = Flask(__name__)
app.config['SECRET_KEY'] = 'change-this-secret-key-123'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# ==================== LOAD MODEL ====================
model_data = joblib.load("model.joblib")
pipeline = model_data["model"]
print("Model loaded → Real risk 3% to 99%")

def predict_stroke_risk(data):
    df = pd.DataFrame([data])
    prob = pipeline.predict_proba(df)[0][1]
    risk = round(prob * 100, 2)
    result = "Likely" if prob >= 0.40 else "Not Likely"
    return result, risk

# ==================== DATABASE MODELS ====================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    user_type = db.Column(db.String(20), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    tests = db.relationship('TestResult', backref='user', lazy=True)

class TestResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    result = db.Column(db.String(20), nullable=False)
    risk_probability = db.Column(db.Float)
    test_date = db.Column(db.DateTime, default=datetime.utcnow)
    gender = db.Column(db.String(10))
    age = db.Column(db.Integer)
    hypertension = db.Column(db.Integer)
    heart_disease = db.Column(db.Integer)
    ever_married = db.Column(db.String(10))
    work_type = db.Column(db.String(20))
    residence_type = db.Column(db.String(20))
    avg_glucose_level = db.Column(db.Float)
    bmi = db.Column(db.Float)
    smoking_status = db.Column(db.String(20))

# ==================== ROUTES ====================

@app.route("/", methods=["GET", "POST"])
def index():
    if 'user_id' not in session:
        return render_template("landing.html")

    error = None
    if request.method == "POST":
        try:
            data = {
                "gender": request.form["gender"].lower(),
                "age": int(request.form["age"]),
                "hypertension": int(request.form["hypertension"]),
                "heart_disease": int(request.form["heart_disease"]),
                "ever_married": request.form["ever_married"].lower(),
                "work_type": request.form["work_type"],
                "Residence_type": request.form["residence_type"].lower(),
                "avg_glucose_level": float(request.form["avg_glucose_level"]),
                "bmi": float(request.form["bmi"]),
                "smoking_status": request.form["smoking_status"].lower(),
            }

            # Fix work type
            work_map = {
                "Government job": "Govt_job", "Never Worked": "Never_worked",
                "Self-employed": "Self-employed", "Children": "children", "Private": "Private"
            }
            data["work_type"] = work_map.get(data["work_type"], data["work_type"])

            # Predict
            result, risk = predict_stroke_risk(data)

            # Save to DB
            save_data = data.copy()
            save_data["residence_type"] = save_data.pop("Residence_type")

            new_test = TestResult(
                user_id=session['user_id'],
                result=result,
                risk_probability=risk,
                **save_data
            )
            db.session.add(new_test)
            db.session.commit()

            return render_template("result.html",
                                   result=result,
                                   probability=risk,
                                   input_data=save_data,
                                   date=datetime.utcnow().strftime("%B %d, %Y"))

        except Exception as e:
            error = "Please fill all fields correctly."
            print("Error:", e)

    return render_template("index.html", error=error)

@app.route("/signup/<user_type>", methods=["GET", "POST"])
def signup(user_type):
    if request.method == "POST":
        email = request.form["email"]
        if User.query.filter_by(email=email).first():
            flash("Email already exists")
            return redirect(url_for('signup', user_type=user_type))
        user = User(
            email=email,
            password=generate_password_hash(request.form["password"]),
            user_type=user_type,
            name=request.form["name"]
        )
        db.session.add(user)
        db.session.commit()
        flash("Account created successfully!")
        return redirect(url_for('login', user_type=user_type))
    return render_template(f"{user_type}_signup.html")

@app.route("/login/<user_type>", methods=["GET", "POST"])
def login(user_type):
    if request.method == "POST":
        user = User.query.filter_by(email=request.form["email"], user_type=user_type).first()
        if user and check_password_hash(user.password, request.form["password"]):
            session['user_id'] = user.id
            session['user_type'] = user_type
            return redirect(url_for('index'))
        flash("Invalid email or password")
    return render_template(f"{user_type}_login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route("/dashboard")
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('index'))

    user = User.query.get(session['user_id'])

    # Doctor Dashboard
    if user.user_type == 'doctor':
        patients = User.query.filter_by(user_type='patient').all()
        data = []
        for p in patients:
            last = TestResult.query.filter_by(user_id=p.id)\
                                   .order_by(TestResult.test_date.desc())\
                                   .first()
            data.append({
                'name': p.name or "Unknown",
                'email': p.email or "—",
                'last_test': last.test_date.strftime("%b %d, %Y") if last else "No assessment",
                'risk': last.result if last and last.result else "Pending"
            })
        return render_template("doctor_dashboard.html", user=user, patients=data)

    # Patient Dashboard
    else:
        latest_test_obj = TestResult.query.filter_by(user_id=user.id)\
                                          .order_by(TestResult.test_date.desc())\
                                          .first()

        latest_test = None
        if latest_test_obj:
            latest_test = {
                'result': latest_test_obj.result or "Unknown",
                'test_date': latest_test_obj.test_date,
                'bmi': float(latest_test_obj.bmi) if latest_test_obj.bmi else 0.0,
                'avg_glucose_level': float(latest_test_obj.avg_glucose_level) if latest_test_obj.avg_glucose_level else 0.0,
                'hypertension': bool(latest_test_obj.hypertension),
                'heart_disease': bool(latest_test_obj.heart_disease),
            }

        history = TestResult.query.filter_by(user_id=user.id)\
                                  .order_by(TestResult.test_date.desc())\
                                  .limit(5).all()

        return render_template("patient_dashboard.html",
                               user=user,
                               latest_test=latest_test,
                               history=history)

@app.route("/test_history")
def test_history():
    if 'user_id' not in session:
        return redirect(url_for('index'))

    user = User.query.get(session['user_id'])
    tests = TestResult.query.filter_by(user_id=user.id)\
                            .order_by(TestResult.test_date.desc())\
                            .all()

    return render_template("test_history.html", user=user, tests=tests)

@app.route("/hospitals")
def hospitals():
    try:
        lat = float(request.args.get("lat"))
        lon = float(request.args.get("lon"))
    except:
        return jsonify([])

    query = (
        "[out:json][timeout:30];"
        "("
        f"node['amenity'='hospital'](around:8000,{lat},{lon});"
        f"way['amenity'='hospital'](around:8000,{lat},{lon});"
        f"node['healthcare'='hospital'](around:8000,{lat},{lon});"
        ");"
        "out center;"
    )

    try:
        response = requests.post(
            "https://overpass-api.de/api/interpreter",
            data={"data": query},
            timeout=20,
            headers={"User-Agent": "StrokeApp/1.0"}
        )
        data = response.json()
    except:
        return jsonify([])

    results = []
    for element in data.get("elements", []):
        tags = element.get("tags", {})
        name = tags.get("name", "Unnamed Hospital")

        if any(word in name.lower() for word in ["eye", "dental", "clinic", "optical", "vision"]):
            continue

        lat2 = element.get("lat") or element.get("center", {}).get("lat")
        lon2 = element.get("lon") or element.get("center", {}).get("lon")
        if not (lat2 and lon2):
            continue

        R = 6371
        dlat = radians(lat2 - lat)
        dlon = radians(lon2 - lon)
        a = sin(dlat/2)**2 + cos(radians(lat)) * cos(radians(lat2)) * sin(dlon/2)**2
        distance = R * 2 * atan2(sqrt(a), sqrt(1-a))

        results.append({
            "name": name,
            "address": tags.get("addr:full") or tags.get("addr:street") or "Address not available",
            "distance": round(distance, 1)
        })

    results.sort(key=lambda x: x["distance"])
    return jsonify(results[:10])

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)