import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'knn_heart_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'heart_scaler.pkl')
COLUMNS_PATH = os.path.join(BASE_DIR, 'heart_columns.pkl')

model = None
scaler = None
expected_columns = None

def load_ml_assets():
    global model, scaler, expected_columns
    print("\n==============================================")
    print("         LOADING ML ASSETS WITH JOBLIB        ")
    print("==============================================")
    
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(COLUMNS_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            expected_columns = joblib.load(COLUMNS_PATH)
            print(" successfully loaded.")
            print(f"Loaded Columns: {expected_columns}")
        except Exception as e:
            print(f" Error during joblib.load: {str(e)}")
    else:
        print(" Error: some files")
        print(f"Checking Path: \n- Model: {MODEL_PATH}\n- Scaler: {SCALER_PATH}\n- Columns: {COLUMNS_PATH}")
    print("==============================================\n")


load_ml_assets()

@app.route('/')
def home():
    
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    if model is None or scaler is None or expected_columns is None:
        return jsonify({'success': False, 'message': 'ML components not loaded.'}), 500

    try:
        
        data = request.get_json(force=True)
        if not data:
            return jsonify({'success': False, 'message': 'Empty data received.'}), 400

        
        sex_map = { "1": "M", "0": "F" }
        cp_map = { "0": "TA", "1": "ATA", "2": "NAP", "3": "ASY" }
        ecg_map = { "0": "Normal", "1": "ST", "2": "LVH" }
        angina_map = { "1": "Y", "0": "N" }
        slope_map = { "0": "Up", "1": "Flat", "2": "Down" }

        sex_val = sex_map.get(str(data.get('sex')).strip(), "M")
        cp_val = cp_map.get(str(data.get('chest_pain')).strip(), "ASY")
        ecg_val = ecg_map.get(str(data.get('resting_ecg')).strip(), "Normal")
        angina_val = angina_map.get(str(data.get('exercise_angina')).strip(), "N")
        slope_val = slope_map.get(str(data.get('st_slope')).strip(), "Flat")

        
        raw_input = {
            'Age': float(data.get('age', 40) or 40),
            'RestingBP': float(data.get('resting_bp', 120) or 120),
            'Cholesterol': float(data.get('cholesterol', 200) or 200),
            'FastingBS': int(data.get('fasting_bs', 0) or 0),
            'MaxHR': float(data.get('max_hr', 150) or 150),
            'Oldpeak': float(data.get('oldpeak', 0.0) or 0.0),
            'Sex_' + sex_val: 1,
            'ChestPainType_' + cp_val: 1,
            'RestingECG_' + ecg_val: 1,
            'ExerciseAngina_' + angina_val: 1,
            'ST_Slope_' + slope_val: 1
        }
        input_df = pd.DataFrame([raw_input])

        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[expected_columns]

        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]

        high_risk_percentage = 0.0
        
        if hasattr(model, "predict_proba"):
            try:
                prob_matrix = model.predict_proba(scaled_input)
                if len(prob_matrix.shape) > 1 and prob_matrix.shape[1] > 1:
                    high_risk_percentage = float(prob_matrix[0][1]) * 100
                else:
                    high_risk_percentage = float(prob_matrix[0]) * 100
            except Exception:
                high_risk_percentage = 85.0 if prediction == 1 else 15.0
        else:
            high_risk_percentage = 85.0 if prediction == 1 else 15.0

        high_risk_percentage = round(high_risk_percentage, 2)
        low_risk_percentage = round(100.0 - high_risk_percentage, 2)

        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'high_risk_prob': high_risk_percentage,
            'low_risk_prob': low_risk_percentage,
            'status': 'High Risk' if prediction == 1 else 'Low Risk'
        })

    except Exception as e:
        import traceback
        print("\n !!! ERROR  IN FLASK BACKEND !!! ")
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Server Error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)