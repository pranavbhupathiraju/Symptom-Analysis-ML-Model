# Flask ML Service Integration

This document describes the expected interface for the Flask ML service that LifeSaver API communicates with.

## Service Endpoint

**URL:** `http://localhost:5000/predict`  
**Method:** `POST`  
**Content-Type:** `application/json`

## Request Format

The Flask service should accept the same JSON format as the Spring Boot API:

```json
{
  "symptoms": ["chest pain", "shortness of breath"],
  "duration": 2,
  "age": 45
}
```

### Field Descriptions

- **symptoms**: Array of strings representing the patient's symptoms
- **duration**: Integer representing the duration of symptoms in days
- **age**: Integer representing the patient's age in years

## Response Format

The Flask service should return a JSON response with the following structure:

```json
{
  "risk_level": "high",
  "suggested_action": "Go to ER immediately",
  "confidence_score": 0.85
}
```

### Field Descriptions

- **risk_level**: String - one of `"low"`, `"moderate"`, or `"high"`
- **suggested_action**: String - one of:
  - `"Monitor at home"`
  - `"Visit urgent care"`
  - `"Go to ER immediately"`
- **confidence_score**: Float between 0.0 and 1.0

## Example Flask Implementation

```python
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('triage_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Extract features
        symptoms = data.get('symptoms', [])
        duration = data.get('duration', 0)
        age = data.get('age', 0)
        
        # Process symptoms (convert to feature vector)
        # This is where you'd implement your feature engineering
        features = process_symptoms(symptoms, duration, age)
        
        # Make prediction
        prediction = model.predict([features])[0]
        confidence = model.predict_proba([features]).max()
        
        # Map prediction to response
        risk_level, suggested_action = map_prediction_to_response(prediction)
        
        return jsonify({
            'risk_level': risk_level,
            'suggested_action': suggested_action,
            'confidence_score': float(confidence)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def process_symptoms(symptoms, duration, age):
    # Implement your feature engineering logic here
    # This is a placeholder - you'll need to implement based on your model
    pass

def map_prediction_to_response(prediction):
    # Map your model's prediction to risk level and action
    # This is a placeholder - implement based on your model's output
    pass

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
```

## Error Handling

The Flask service should handle errors gracefully and return appropriate HTTP status codes:

- **200 OK**: Successful prediction
- **400 Bad Request**: Invalid input data
- **500 Internal Server Error**: Model or processing error

## Testing the Integration

Once both services are running, you can test the integration:

```bash
# Test the Spring Boot API
curl -X POST http://localhost:8080/api/v1/triage \
  -H "Content-Type: application/json" \
  -d '{
    "symptoms": ["chest pain", "shortness of breath"],
    "duration": 2,
    "age": 45
  }'
```

This should return a response like:
```json
{
  "risk_level": "high",
  "suggested_action": "Go to ER immediately",
  "confidence_score": 0.85
}
``` 