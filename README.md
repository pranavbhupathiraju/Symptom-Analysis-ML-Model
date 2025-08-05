# Health Triage API

A RESTful API for emergency health triage that uses machine learning to assess risk levels and provide medical recommendations.

## What it does

This project combines a Spring Boot backend with a Python Flask microservice to create an emergency health triage system. Users can input symptoms, duration, and age, and the system will return a risk assessment (low/moderate/high) along with recommended actions.

## How it works

- **Spring Boot API**: Handles HTTP requests and validation
- **Flask ML Service**: Runs the machine learning model
- **RandomForest Classifier**: Trained on medical data with 91.5% accuracy
- **Simple HTTP communication**: Uses RestTemplate for service-to-service calls

## Tech Stack

**Backend:**
- Java 17 + Spring Boot 3.2.0
- Spring Web MVC for REST endpoints
- RestTemplate for HTTP communication
- Maven for build management

**ML Service:**
- Python 3.12 + Flask 3.0.0
- Scikit-learn for machine learning
- Joblib for model persistence


### Setup

1. **Start the Flask ML Service:**

cd flask_service
pip install -r requirements.txt
python3 app_realistic.py


2. **Start the Spring Boot API:**

mvn spring-boot:run


3. **Test the API:**

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


## API Endpoints

### GET /api/v1/symptoms
Returns all supported symptoms.

**Response:**

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


### POST /api/v1/triage
Performs health triage assessment.

**Request:**

{
  "symptoms": ["chest pain", "shortness of breath", "nausea"],
  "duration": 1,
  "age": 65
}


**Response:**

{
  "risk_level": "high",
  "suggested_action": "Go to ER immediately",
  "confidence_score": 0.68
}


## Risk Levels
1. low -> Monitor at home -> Self-care recommended 
2. moderate -> Visit urgent care -> Professional evaluation needed 
3. high -> Go to ER immediately -> Emergency medical attention required 

## Configuration

The application uses `application.yml` for configuration:


server:
  port: 8080

ml:
  service:
    url: http://localhost:5001

logging:
  level:
    com.healthtriage: DEBUG



## Development Instructions

### Building
`
mvn clean compile
mvn test
mvn package


### Training the ML Model

cd flask_service
python3 train_model.py


## Testing

### Manual Testing

# Test Flask service directly
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"symptoms": ["fever", "cough"], "duration": 3, "age": 35}'

# Test Spring Boot API
curl -X POST http://localhost:8080/api/v1/triage \
  -H "Content-Type: application/json" \
  -d '{"symptoms": ["chest pain"], "duration": 1, "age": 65}'


### Automated Testing

mvn test
cd flask_service && python3 test_realistic_model.py



## License

MIT License

---

Built with Spring Boot and Flask.
