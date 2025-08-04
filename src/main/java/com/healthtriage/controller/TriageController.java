package com.healthtriage.controller;

import com.healthtriage.dto.TriageRequest;
import com.healthtriage.dto.TriageResponse;
import com.healthtriage.service.TriageService;
import jakarta.validation.Valid;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import reactor.core.publisher.Mono;

import java.util.Arrays;
import java.util.List;

@RestController
@RequestMapping("/api/v1")
@CrossOrigin(origins = "*")
public class TriageController {
    
    private static final Logger logger = LoggerFactory.getLogger(TriageController.class);
    
    private final TriageService triageService;
    
    // Supported symptoms list
    private static final List<String> SUPPORTED_SYMPTOMS = Arrays.asList(
        "chest pain", "shortness of breath", "fever", "cough", "headache",
        "dizziness", "nausea", "vomiting", "abdominal pain", "fatigue",
        "muscle aches", "sore throat", "runny nose", "sneezing", "itchy eyes",
        "paralysis", "vision problems", "speech problems", "loss of appetite",
        "unconsciousness", "confusion", "excessive thirst", "blurred vision",
        "sensitivity to light", "diarrhea", "sleep problems", "anxiety", "mild fever"
    );
    
    public TriageController(TriageService triageService) {
        this.triageService = triageService;
    }
    
    @GetMapping("/symptoms")
    public ResponseEntity<List<String>> getSupportedSymptoms() {
        logger.info("Returning supported symptoms list");
        return ResponseEntity.ok(SUPPORTED_SYMPTOMS);
    }
    
    @PostMapping("/triage")
    public Mono<ResponseEntity<TriageResponse>> performTriage(@Valid @RequestBody TriageRequest request) {
        logger.info("Received triage request: {}", request);
        
        return triageService.getTriagePrediction(request)
                .map(response -> {
                    logger.info("Triage completed successfully: {}", response);
                    return ResponseEntity.ok(response);
                })
                .onErrorResume(RuntimeException.class, error -> {
                    logger.error("Triage failed: {}", error.getMessage());
                    return Mono.just(ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE)
                            .body(new TriageResponse("unknown", "Service temporarily unavailable", 0.0)));
                });
    }
    
    @ExceptionHandler(Exception.class)
    public ResponseEntity<String> handleGenericException(Exception ex) {
        logger.error("Unexpected error in controller: {}", ex.getMessage());
        return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                .body("An unexpected error occurred. Please try again later.");
    }
} 