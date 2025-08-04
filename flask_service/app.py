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


def load_model():
    """Load the trained model and label binarizer"""
    global model, label_binarizer
    try:
        model = joblib.load('model.pkl')
        label_binarizer = joblib.load('label_binarizer.pkl')
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def prepare_features(symptoms, duration, age):
    """Prepare features for prediction"""
    # One-hot encode symptoms
    symptoms_encoded = label_binarizer.transform([symptoms])

    # Combine with duration and age
    features = np.concatenate([symptoms_encoded, [[duration, age]]], axis=1)

    return features


@app.route('/predict', methods=['POST'])
def predict():
    """Predict risk level and suggested action"""
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

        # Prepare response
        response = {
            'risk_level': prediction,
            'suggested_action': suggested_action,
            'confidence_score': confidence
        }

        logger.info(f"Prediction: {response}")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})


@app.route('/', methods=['GET'])
def root():
    """Root endpoint with API information"""
    return jsonify({
        'service': 'LifeSaver ML Service',
        'version': '1.0.0',
        'endpoints': {
            'POST /predict': 'Predict risk level and action',
            'GET /health': 'Health check',
            'GET /': 'This information'
        }
    })


if __name__ == '__main__':
    # Load model on startup
    load_model()

    # Run the Flask app on port 5001
    app.run(host='0.0.0.0', port=5001, debug=True)
