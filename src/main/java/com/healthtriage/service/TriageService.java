package com.healthtriage.service;

import com.healthtriage.dto.TriageRequest;
import com.healthtriage.dto.TriageResponse;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;

@Service
public class TriageService {
    private static final Logger logger = LoggerFactory.getLogger(TriageService.class);
    private final RestTemplate restTemplate;
    
    @Value("${ml.service.url:http://localhost:5001}")
    private String mlServiceUrl;

    public TriageService(RestTemplate restTemplate) {
        this.restTemplate = restTemplate;
    }

    public TriageResponse getTriagePrediction(TriageRequest request) {
        logger.info("Received triage request: {}", request);
        logger.info("Sending triage request to ML service: {}", request);
        
        try {
            // Set up headers
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            
            // Create request entity
            HttpEntity<TriageRequest> requestEntity = new HttpEntity<>(request, headers);
            
            // Make the API call
            TriageResponse response = restTemplate.postForObject(
                mlServiceUrl + "/predict", 
                requestEntity, 
                TriageResponse.class
            );
            
            logger.info("Received triage response: {}", response);
            logger.info("Triage completed successfully: {}", response);
            
            return response;
            
        } catch (Exception e) {
            logger.error("Error calling ML service: {}", e.getMessage());
            throw new RuntimeException("ML service is currently unavailable. Please try again later.", e);
        }
    }
} 