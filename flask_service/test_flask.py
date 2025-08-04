import joblib
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

# Load the model and label binarizer
print("Loading model...")
model = joblib.load('model.pkl')
label_binarizer = joblib.load('label_binarizer.pkl')
print("Model loaded successfully!")

# Test prediction
test_symptoms = ["chest pain", "shortness of breath"]
test_duration = 2
test_age = 45

print(
    f"Testing with symptoms: {test_symptoms}, duration: {test_duration}, age: {test_age}")

# Prepare features
symptoms_encoded = label_binarizer.transform([test_symptoms])
features = np.concatenate(
    [symptoms_encoded, [[test_duration, test_age]]], axis=1)

# Make prediction
prediction = model.predict(features)[0]
confidence_scores = model.predict_proba(features)[0]
confidence = float(np.max(confidence_scores))

print(f"Prediction: {prediction}")
print(f"Confidence: {confidence}")

# Test with different symptoms
test_symptoms2 = ["fever", "cough"]
symptoms_encoded2 = label_binarizer.transform([test_symptoms2])
features2 = np.concatenate([symptoms_encoded2, [[1, 30]]], axis=1)
prediction2 = model.predict(features2)[0]
confidence2 = float(np.max(model.predict_proba(features2)[0]))

print(
    f"Test 2 - Symptoms: {test_symptoms2}, Prediction: {prediction2}, Confidence: {confidence2}")
