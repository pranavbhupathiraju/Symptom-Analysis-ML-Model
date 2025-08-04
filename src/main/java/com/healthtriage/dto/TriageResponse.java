package com.healthtriage.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.Objects;

public class TriageResponse {
    @JsonProperty("risk_level")
    private String riskLevel;

    @JsonProperty("suggested_action")
    private String suggestedAction;

    @JsonProperty("confidence_score")
    private Double confidenceScore;

    // Default constructor
    public TriageResponse() {}

    // Constructor with all fields
    public TriageResponse(String riskLevel, String suggestedAction, Double confidenceScore) {
        this.riskLevel = riskLevel;
        this.suggestedAction = suggestedAction;
        this.confidenceScore = confidenceScore;
    }

    // Getters and Setters
    public String getRiskLevel() {
        return riskLevel;
    }

    public void setRiskLevel(String riskLevel) {
        this.riskLevel = riskLevel;
    }

    public String getSuggestedAction() {
        return suggestedAction;
    }

    public void setSuggestedAction(String suggestedAction) {
        this.suggestedAction = suggestedAction;
    }

    public Double getConfidenceScore() {
        return confidenceScore;
    }

    public void setConfidenceScore(Double confidenceScore) {
        this.confidenceScore = confidenceScore;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        TriageResponse that = (TriageResponse) o;
        return Objects.equals(riskLevel, that.riskLevel) &&
                Objects.equals(suggestedAction, that.suggestedAction) &&
                Objects.equals(confidenceScore, that.confidenceScore);
    }

    @Override
    public int hashCode() {
        return Objects.hash(riskLevel, suggestedAction, confidenceScore);
    }

    @Override
    public String toString() {
        return "TriageResponse{" +
                "riskLevel='" + riskLevel + '\'' +
                ", suggestedAction='" + suggestedAction + '\'' +
                ", confidenceScore=" + confidenceScore +
                '}';
    }
} 