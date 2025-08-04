package com.healthtriage.service;

import com.healthtriage.dto.TriageRequest;
import com.healthtriage.dto.TriageResponse;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.client.WebClientResponseException;
import reactor.core.publisher.Mono;

@Service
public class TriageService {
    private static final Logger logger = LoggerFactory.getLogger(TriageService.class);
    private final WebClient webClient;
    
    @Value("${ml.service.url:http://localhost:5001}")
    private String mlServiceUrl;

    public TriageService(WebClient.Builder webClientBuilder) {
        this.webClient = webClientBuilder.build();
    }

    public Mono<TriageResponse> getTriagePrediction(TriageRequest request) {
        logger.info("Received triage request: {}", request);
        logger.info("Sending triage request to ML service: {}", request);
        
        return webClient.post()
                .uri(mlServiceUrl + "/predict")
                .bodyValue(request)
                .retrieve()
                .bodyToMono(TriageResponse.class)
                .doOnSuccess(response -> logger.info("Received triage response: {}", response))
                .doOnSuccess(response -> logger.info("Triage completed successfully: {}", response))
                .onErrorMap(WebClientResponseException.class, this::handleWebClientException)
                .onErrorMap(Exception.class, this::handleGenericException);
    }

    private RuntimeException handleWebClientException(WebClientResponseException ex) {
        logger.error("WebClient error: {} - {}", ex.getStatusCode(), ex.getResponseBodyAsString());
        return new RuntimeException("ML service communication error: " + ex.getMessage());
    }

    private RuntimeException handleGenericException(Exception ex) {
        logger.error("Unexpected error in service: {}", ex.getMessage(), ex);
        return new RuntimeException("Service error: " + ex.getMessage());
    }
} 