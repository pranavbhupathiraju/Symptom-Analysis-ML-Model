Health Triage API: ML-Powered Health Assessment
Ever wondered if a machine could help make an initial health assessment? This project is a RESTful API I built for that exact purpose. It uses a machine learning model to quickly assess a user's risk level based on their symptoms, providing immediate, actionable recommendations.

The Challenge
The goal was to build a system that could quickly and accurately triage health symptoms. I decided to tackle this by integrating a robust machine learning model with a user-friendly API, allowing for a fast and reliable way to get an initial risk assessment without the need for a human on the front end.

My Approach
I chose a microservice architecture to keep the backend separate from the ML logic. This approach allows for independent scaling and development.

Spring Boot: The core API handles all incoming requests and validation, acting as the main interface.

Flask Microservice: This service is a dedicated home for the machine learning model, a RandomForest Classifier I trained on medical data to achieve 91.5% accuracy.

Inter-Service Communication: The two services talk to each other via a simple HTTP connection using RestTemplate. This keeps the system lightweight and easy to maintain.

Tech Stack
Backend:

Java 17 + Spring Boot 3.2.0

Spring Web MVC for REST endpoints

RestTemplate for HTTP communication

Maven for build management

ML Service:

Python 3.12 + Flask 3.0.0

Scikit-learn for machine learning

Joblib for model persistence

Getting Started
1. Run the Flask ML Service First:

cd flask_service
pip install -r requirements.txt
python3 app_realistic.py
2. Start the Spring Boot API:

mvn spring-boot:run
3. Test the API:

# Get symptoms list
curl http://localhost:8080/api/v1/symptoms

# Perform triage
curl -X POST http://localhost:8080/api/v1/triage \
  -H "Content-Type: application/json" \
  -d '{
    "symptoms": ["chest pain", "shortness of breath"],
    "duration": 1,
    "age": 65
  }'
API Endpoints
GET /api/v1/symptoms
Returns all supported symptoms.

Response:

JSON

[
  "chest pain",
  "shortness of breath", 
  "fever",
  "cough",
  "headache",
  "dizziness",
  "nausea",
  "vomiting",
  "abdominal pain",
  "fatigue",
  "muscle aches",
  "sore throat",
  "runny nose",
  "sneezing",
  "itchy eyes",
  "paralysis",
  "vision problems",
  "speech problems",
  "loss of appetite",
  "unconsciousness",
  "confusion",
  "excessive thirst",
  "blurred vision",
  "sensitivity to light",
  "diarrhea",
  "sleep problems",
  "anxiety",
  "mild fever"
]
POST /api/v1/triage
Performs health triage assessment.

Request:

JSON

{
  "symptoms": ["chest pain", "shortness of breath", "nausea"],
  "duration": 1,
  "age": 65
}
Response:

JSON

{
  "risk_level": "high",
  "suggested_action": "Go to ER immediately",
  "confidence_score": 0.68
}
Risk Levels
low -> Monitor at home -> Self-care recommended 

moderate -> Visit urgent care -> Professional evaluation needed 

high -> Go to ER immediately -> Emergency medical attention required 

Configuration
The application uses application.yml for configuration:

YAML

server:
  port: 8080

ml:
  service:
    url: http://localhost:5001

logging:
  level:
    com.healthtriage: DEBUG
Development Instructions
Building
mvn clean compile
mvn test
mvn package
Training the ML Model
cd flask_service
python3 train_model.py
Testing
Manual Testing
# Test Flask service directly
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"symptoms": ["fever", "cough"], "duration": 3, "age": 35}'

# Test Spring Boot API
curl -X POST http://localhost:8080/api/v1/triage \
  -H "Content-Type: application/json" \
  -d '{"symptoms": ["chest pain"], "duration": 1, "age": 65}'
Automated Testing
mvn test
cd flask_service && python3 test_realistic_model.py
License
MIT License
