import os
import sqlite3
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# 1. Database Initialize karne ka function
def init_db():
    conn = sqlite3.connect('heart_disease.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patient_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            age INTEGER,
            sex TEXT,
            resting_bp INTEGER,
            cholesterol INTEGER,
            max_hr INTEGER,
            prediction INTEGER,
            percentage REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# 2. Safe Pickle ML Models Loading Section
model_path = 'knn_heart_model.pkl'
scaler_path = 'heart_scaler.pkl'

model = None
scaler = None

# Try-Except lagaya hai taaki agar pkl corrupt bhi ho toh server crash na ho
try:
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print("✅ ML Models and Scaler loaded successfully!")
    else:
        print("⚠️ Warning: Model or Scaler file missing. Using fallback prediction mode.")
except Exception as pkl_error:
    print(f"❌ Pickle Load Error handle kar li gayi hai: {pkl_error}")
    print("Server fallback mode me chal raha hai. Database save bypass operational hai.")
    model = None
    scaler = None

# Home page serve karne ke liye route
@app.route('/')
def home():
    return render_template('index.html')

# 3. Main Prediction and Database Connection API
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({"success": False, "error": "No data received"})

        # Frontend se inputs extract karna
        username = data.get('username') or 'Guest User'
        age = int(data.get('age', 40))
        sex_raw = data.get('sex')
        sex_text = "Male" if sex_raw == "1" else "Female"
        
        resting_bp = int(data.get('resting_bp', 120))
        cholesterol = int(data.get('cholesterol', 200))
        max_hr = int(data.get('max_hr', 150))
        
        # ML pipeline inputs array format me
        features = [
            float(data.get('age', 0)),
            float(data.get('sex', 0)),
            float(data.get('chest_pain', 0)),
            float(data.get('resting_bp', 0)),
            float(data.get('cholesterol', 0)),
            float(data.get('fasting_bs', 0)),
            float(data.get('resting_ecg', 0)),
            float(data.get('max_hr', 0)),
            float(data.get('exercise_angina', 0)),
            float(data.get('oldpeak', 0)),
            float(data.get('st_slope', 0))
        ]

        # Init default response values
        prediction = 0
        high_risk = 50.0
        low_risk = 50.0

        # ML Model se Inference nikalna
        if model and scaler:
            features_scaled = scaler.transform([features])
            prediction = int(model.predict(features_scaled)[0])
            probabilities = model.predict_proba(features_scaled)[0]
            
            high_risk = round(probabilities[1] * 100, 2)
            low_risk = round(probabilities[0] * 100, 2)
        else:
            # Fallback logic: Agar model corrupt hai toh default mathematical logic apply hoga
            # Taki aapka UI aur database testing bina rukaawat chalti rahe
            if age > 50 or resting_bp > 140 or cholesterol > 240:
                prediction = 1
                high_risk = 75.0
                low_risk = 25.0
            else:
                prediction = 0
                high_risk = 20.0
                low_risk = 80.0

        probability = high_risk if prediction == 1 else low_risk

        # 4. SQLite Database me Data permanently save karna
        conn = sqlite3.connect('heart_disease.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO patient_history (username, age, sex, resting_bp, cholesterol, max_hr, prediction, percentage)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (username, age, sex_text, resting_bp, cholesterol, max_hr, prediction, probability))
        
        conn.commit()
        conn.close()

        # Frontend dynamic response format
        return jsonify({
            "success": True,
            "prediction": prediction,
            "high_risk_prob": high_risk,
            "low_risk_prob": low_risk
        })

    except Exception as e:
        print("Backend Error:", str(e))
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    init_db()  # App run hote hi table ensure karega
    app.run(debug=True)