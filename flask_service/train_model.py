import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import joblib
import random

# Define symptoms that match the Spring Boot API
SYMPTOMS = [
    "chest pain", "shortness of breath", "fever", "cough", "headache",
    "dizziness", "nausea", "vomiting", "abdominal pain", "fatigue",
    "muscle aches", "sore throat", "runny nose", "diarrhea", "constipation",
    "back pain", "joint pain", "swelling", "rash", "bleeding",
    "unconsciousness", "seizure", "paralysis", "vision problems", "hearing problems"
]

# Risk levels and actions
RISK_LEVELS = ["low", "moderate", "high"]
ACTIONS = {
    "low": "Monitor at home",
    "moderate": "Visit urgent care",
    "high": "Go to ER immediately"
}


def generate_synthetic_data(n_samples=1000):
    """Generate synthetic training data"""
    data = []

    for _ in range(n_samples):
        # Random age between 1 and 100
        age = random.randint(1, 100)

        # Random duration between 0 and 30 days
        duration = random.randint(0, 30)

        # Random number of symptoms (1-5)
        num_symptoms = random.randint(1, 5)
        symptoms = random.sample(SYMPTOMS, num_symptoms)

        # Determine risk level based on symptoms and age
        risk_level = determine_risk_level(symptoms, age, duration)

        data.append({
            'symptoms': symptoms,
            'duration': duration,
            'age': age,
            'risk_level': risk_level
        })

    return data


def determine_risk_level(symptoms, age, duration):
    """Determine risk level based on symptoms, age, and duration"""

    # High risk symptoms
    high_risk_symptoms = ["chest pain", "shortness of breath",
                          "unconsciousness", "seizure", "paralysis"]

    # Moderate risk symptoms
    moderate_risk_symptoms = ["fever", "abdominal pain",
                              "bleeding", "vision problems", "hearing problems"]

    # Check for high risk combinations
    if any(symptom in symptoms for symptom in high_risk_symptoms):
        return "high"

    # Check for moderate risk combinations
    if any(symptom in symptoms for symptom in moderate_risk_symptoms):
        return "moderate"

    # Age factor
    if age > 65:
        if len(symptoms) >= 3:
            return "moderate"

    # Duration factor
    if duration > 7:
        if len(symptoms) >= 2:
            return "moderate"

    # Default to low risk
    return "low"


def prepare_features(data):
    """Prepare features for training"""
    # One-hot encode symptoms
    mlb = MultiLabelBinarizer()
    symptoms_encoded = mlb.fit_transform([item['symptoms'] for item in data])

    # Create feature matrix
    features = []
    for i, item in enumerate(data):
        feature_vector = list(
            symptoms_encoded[i]) + [item['duration'], item['age']]
        features.append(feature_vector)

    return np.array(features), mlb


def train_model():
    """Train the RandomForest model"""
    print("Generating synthetic training data...")
    data = generate_synthetic_data(2000)

    print("Preparing features...")
    X, mlb = prepare_features(data)
    y = [item['risk_level'] for item in data]

    print(f"Training data shape: {X.shape}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Risk level distribution: {pd.Series(y).value_counts().to_dict()}")

    # Train RandomForest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Save model and label binarizer
    joblib.dump(model, 'model.pkl')
    joblib.dump(mlb, 'label_binarizer.pkl')

    print("Model trained and saved successfully!")
    print(f"Model accuracy: {model.score(X, y):.3f}")

    return model, mlb


if __name__ == "__main__":
    train_model()
