# Health Triage API

A RESTful API for emergency health triage that uses machine learning to assess risk levels and provide medical recommendations.

## What it does

This project combines a Spring Boot backend with a Python Flask microservice to create an emergency health triage system. Users can input symptoms, duration, and age, and the system will return a risk assessment (low/moderate/high) along with recommended actions.

## How it works

- **Spring Boot API** (Port 8080): Handles HTTP requests and validation
- **Flask ML Service** (Port 5001): Runs the machine learning model
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

## Getting Started

### Prerequisites
- Java 17+
- Python 3.8+
- Maven 3.6+

### Setup

1. **Start the Flask ML Service:**
```bash
cd flask_service
pip install -r requirements.txt
python3 app_realistic.py
```

2. **Start the Spring Boot API:**
```bash
mvn spring-boot:run
```

3. **Test the API:**
```bash
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
```

## API Endpoints

### GET /api/v1/symptoms
Returns all supported symptoms.

**Response:**
```json
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
```

### POST /api/v1/triage
Performs health triage assessment.

**Request:**
```json
{
  "symptoms": ["chest pain", "shortness of breath", "nausea"],
  "duration": 1,
  "age": 65
}
```

**Response:**
```json
{
  "risk_level": "high",
  "suggested_action": "Go to ER immediately",
  "confidence_score": 0.68
}
```

## Risk Levels

| Risk Level | Action | Description |
|------------|--------|-------------|
| low | Monitor at home | Self-care recommended |
| moderate | Visit urgent care | Professional evaluation needed |
| high | Go to ER immediately | Emergency medical attention required |

## Configuration

The application uses `application.yml` for configuration:

```yaml
server:
  port: 8080

ml:
  service:
    url: http://localhost:5001

logging:
  level:
    com.healthtriage: DEBUG
```

## Project Structure

```
Health-Triage-API/
├── src/main/java/com/healthtriage/
│   ├── HealthTriageApplication.java
│   ├── controller/
│   │   └── TriageController.java
│   ├── service/
│   │   └── TriageService.java
│   ├── dto/
│   │   ├── TriageRequest.java
│   │   └── TriageResponse.java
│   └── config/
│       └── RestTemplateConfig.java
├── src/main/resources/
│   └── application.yml
├── flask_service/
│   ├── app_realistic.py
│   ├── train_model.py
│   ├── requirements.txt
│   └── test_realistic_model.py
├── pom.xml
└── README.md
```

## Development

### Building
```bash
mvn clean compile
mvn test
mvn package
```

### Training the ML Model
```bash
cd flask_service
python3 train_model.py
```

## Testing

### Manual Testing
```bash
# Test Flask service directly
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"symptoms": ["fever", "cough"], "duration": 3, "age": 35}'

# Test Spring Boot API
curl -X POST http://localhost:8080/api/v1/triage \
  -H "Content-Type: application/json" \
  -d '{"symptoms": ["chest pain"], "duration": 1, "age": 65}'
```

### Automated Testing
```bash
mvn test
cd flask_service && python3 test_realistic_model.py
```

## Error Handling

The API includes basic error handling:

- **Input validation**: Returns 400 for invalid input
- **ML service unavailable**: Returns 503 with fallback response
- **Generic errors**: Returns 500 with error message

## Performance

- API response time: < 200ms
- Model accuracy: 91.5%
- Supports 12 medical conditions
- Simple synchronous HTTP calls

## Security

- Input validation and sanitization
- CORS enabled for web clients
- Error message sanitization
- No sensitive data logging

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License

---

Built with Spring Boot and Flask for emergency medical decision support. 