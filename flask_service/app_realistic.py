from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global variables for model and label binarizer
model = None
label_binarizer = None

# Risk levels and actions mapping
RISK_LEVELS = ["low", "moderate", "high"]
ACTIONS = {
    "low": "Monitor at home",
    "moderate": "Visit urgent care",
    "high": "Go to ER immediately"
}

# Medical conditions mapping for more detailed responses
MEDICAL_CONDITIONS = {
    'heart_attack': 'Possible heart attack - requires immediate emergency care',
    'stroke': 'Possible stroke - requires immediate emergency care',
    'pneumonia': 'Possible pneumonia - requires urgent medical attention',
    'appendicitis': 'Possible appendicitis - requires urgent medical attention',
    'diabetes_emergency': 'Possible diabetic emergency - requires urgent medical attention',
    'flu': 'Influenza-like illness - consider urgent care if severe',
    'migraine': 'Migraine headache - consider urgent care if severe',
    'gastroenteritis': 'Gastroenteritis - consider urgent care if severe',
    'hypertension': 'Hypertension symptoms - consider urgent care',
    'common_cold': 'Common cold - monitor at home',
    'allergies': 'Allergic reaction - monitor at home',
    'stress': 'Stress-related symptoms - monitor at home'
}


def load_model():
    """Load the trained realistic model and label binarizer"""
    global model, label_binarizer
    try:
        model = joblib.load('model_realistic.pkl')
        label_binarizer = joblib.load('label_binarizer_realistic.pkl')
        logger.info("Realistic model loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading realistic model: {e}")
        # Fallback to original model
        try:
            model = joblib.load('model.pkl')
            label_binarizer = joblib.load('label_binarizer.pkl')
            logger.info("Fallback to original model")
        except Exception as e2:
            logger.error(f"Error loading fallback model: {e2}")
            raise


def prepare_features(symptoms, duration, age):
    """Prepare features for prediction"""
    # One-hot encode symptoms
    symptoms_encoded = label_binarizer.transform([symptoms])

    # Combine with duration and age
    features = np.concatenate([symptoms_encoded, [[duration, age]]], axis=1)

    return features


def get_medical_advice(symptoms, risk_level, confidence, duration):
    """Get detailed medical advice based on symptoms and risk level"""
    advice = ""

    # High risk symptoms that require immediate attention
    high_risk_symptoms = ["chest pain", "shortness of breath",
                          "unconsciousness", "seizure", "paralysis"]
    if any(symptom in symptoms for symptom in high_risk_symptoms):
        advice += "⚠️ EMERGENCY: These symptoms require immediate medical attention. "

    # Moderate risk symptoms
    moderate_risk_symptoms = [
        "fever", "abdominal pain", "bleeding", "vision problems"]
    if any(symptom in symptoms for symptom in moderate_risk_symptoms):
        advice += "⚠️ URGENT: These symptoms should be evaluated by a healthcare provider. "

    # Age-specific advice
    if any(symptom in symptoms for symptom in ["chest pain", "shortness of breath", "dizziness"]):
        advice += "🚨 If you are over 65 or have heart conditions, seek immediate care. "

    # Duration advice
    if duration > 7:
        advice += "⏰ Symptoms lasting more than a week should be evaluated. "

    # Confidence-based advice
    if confidence < 0.7:
        advice += "❓ Low confidence prediction - consider consulting a healthcare provider. "

    return advice


@app.route('/predict', methods=['POST'])
def predict():
    """Predict risk level and suggested action with detailed medical advice"""
    try:
        # Get JSON data
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        # Extract features
        symptoms = data.get('symptoms', [])
        # Handle both 'duration' and 'duration_days'
        duration = data.get('duration', 0)
        if 'duration_days' in data:
            duration = data.get('duration_days', 0)
        age = data.get('age', 0)

        # Validate inputs
        if not symptoms or not isinstance(symptoms, list):
            return jsonify({'error': 'Symptoms must be a non-empty list'}), 400

        if not isinstance(duration, (int, float)) or duration < 0:
            return jsonify({'error': 'Duration must be a non-negative number'}), 400

        if not isinstance(age, (int, float)) or age < 0:
            return jsonify({'error': 'Age must be a non-negative number'}), 400

        logger.info(
            f"Received prediction request: symptoms={symptoms}, duration={duration}, age={age}")

        # Prepare features
        features = prepare_features(symptoms, duration, age)

        # Make prediction
        prediction = model.predict(features)[0]
        confidence_scores = model.predict_proba(features)[0]
        confidence = float(np.max(confidence_scores))

        # Get suggested action
        suggested_action = ACTIONS.get(prediction, "Monitor at home")

        # Get detailed medical advice
        medical_advice = get_medical_advice(
            symptoms, prediction, confidence, duration)

        # Prepare response
        response = {
            'risk_level': prediction,
            'suggested_action': suggested_action,
            'confidence_score': confidence,
            'medical_advice': medical_advice,
            'symptoms_analyzed': symptoms,
            'patient_age': age,
            'symptom_duration_days': duration
        }

        logger.info(f"Prediction: {response}")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_type': 'realistic_medical_dataset'
    })


@app.route('/', methods=['GET'])
def root():
    """Root endpoint with API information"""
    return jsonify({
        'service': 'LifeSaver ML Service (Realistic Medical Model)',
        'version': '2.0.0',
        'model_type': 'Realistic medical dataset based on common conditions',
        'endpoints': {
            'POST /predict': 'Predict risk level and action with medical advice',
            'GET /health': 'Health check',
            'GET /': 'This information'
        },
        'supported_conditions': list(MEDICAL_CONDITIONS.keys())
    })


@app.route('/conditions', methods=['GET'])
def get_conditions():
    """Get supported medical conditions"""
    return jsonify({
        'medical_conditions': MEDICAL_CONDITIONS,
        'risk_levels': RISK_LEVELS,
        'actions': ACTIONS
    })


if __name__ == '__main__':
    # Load model on startup
    load_model()

    # Run the Flask app on port 5001
    app.run(host='0.0.0.0', port=5001, debug=True)
