import joblib
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

# Load the realistic model and label binarizer
print("Loading realistic model...")
model = joblib.load('model_realistic.pkl')
label_binarizer = joblib.load('label_binarizer_realistic.pkl')
print("Realistic model loaded successfully!")

# Test cases based on real medical conditions
test_cases = [
    {
        'symptoms': ['chest pain', 'shortness of breath', 'nausea'],
        'duration': 1,
        'age': 65,
        'expected': 'high'
    },
    {
        'symptoms': ['fever', 'cough', 'muscle aches'],
        'duration': 3,
        'age': 35,
        'expected': 'moderate'
    },
    {
        'symptoms': ['runny nose', 'sneezing', 'itchy eyes'],
        'duration': 2,
        'age': 25,
        'expected': 'low'
    },
    {
        'symptoms': ['abdominal pain', 'nausea', 'vomiting'],
        'duration': 1,
        'age': 40,
        'expected': 'high'
    },
    {
        'symptoms': ['headache', 'fatigue', 'muscle aches'],
        'duration': 5,
        'age': 30,
        'expected': 'low'
    }
]

print("\nTesting realistic model with medical scenarios:")
print("=" * 60)

for i, case in enumerate(test_cases, 1):
    symptoms = case['symptoms']
    duration = case['duration']
    age = case['age']
    expected = case['expected']

    # Prepare features
    symptoms_encoded = label_binarizer.transform([symptoms])
    features = np.concatenate([symptoms_encoded, [[duration, age]]], axis=1)

    # Make prediction
    prediction = model.predict(features)[0]
    confidence_scores = model.predict_proba(features)[0]
    confidence = float(np.max(confidence_scores))

    # Determine action
    actions = {
        "low": "Monitor at home",
        "moderate": "Visit urgent care",
        "high": "Go to ER immediately"
    }
    action = actions.get(prediction, "Monitor at home")

    print(f"\nTest Case {i}:")
    print(f"  Symptoms: {symptoms}")
    print(f"  Duration: {duration} days, Age: {age}")
    print(f"  Expected: {expected}")
    print(f"  Predicted: {prediction} (confidence: {confidence:.2f})")
    print(f"  Action: {action}")
    print(f"  {'✅' if prediction == expected else '❌'} Match")

print(f"\nModel Performance Summary:")
print(f"  Total tests: {len(test_cases)}")
print(
    f"  Correct predictions: {sum(1 for case in test_cases if case['expected'] == model.predict(np.concatenate([label_binarizer.transform([case['symptoms']]), [[case['duration'], case['age']]]], axis=1))[0])}")
print(
    f"  Accuracy: {sum(1 for case in test_cases if case['expected'] == model.predict(np.concatenate([label_binarizer.transform([case['symptoms']]), [[case['duration'], case['age']]]], axis=1))[0]) / len(test_cases):.2f}")
