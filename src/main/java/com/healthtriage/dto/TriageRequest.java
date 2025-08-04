package com.healthtriage.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.NotNull;

import java.util.List;
import java.util.Objects;

public class TriageRequest {
    @NotEmpty(message = "Symptoms list cannot be empty")
    @JsonProperty("symptoms")
    private List<String> symptoms;

    @NotNull(message = "Duration is required")
    @Min(value = 0, message = "Duration must be non-negative")
    @JsonProperty("duration")
    private Integer duration;

    @NotNull(message = "Age is required")
    @Min(value = 0, message = "Age must be non-negative")
    @JsonProperty("age")
    private Integer age;

    // Default constructor
    public TriageRequest() {}

    // Constructor with all fields
    public TriageRequest(List<String> symptoms, Integer duration, Integer age) {
        this.symptoms = symptoms;
        this.duration = duration;
        this.age = age;
    }

    // Getters and Setters
    public List<String> getSymptoms() {
        return symptoms;
    }

    public void setSymptoms(List<String> symptoms) {
        this.symptoms = symptoms;
    }

    public Integer getDuration() {
        return duration;
    }

    public void setDuration(Integer duration) {
        this.duration = duration;
    }

    public Integer getAge() {
        return age;
    }

    public void setAge(Integer age) {
        this.age = age;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        TriageRequest that = (TriageRequest) o;
        return Objects.equals(symptoms, that.symptoms) &&
                Objects.equals(duration, that.duration) &&
                Objects.equals(age, that.age);
    }

    @Override
    public int hashCode() {
        return Objects.hash(symptoms, duration, age);
    }

    @Override
    public String toString() {
        return "TriageRequest{" +
                "symptoms=" + symptoms +
                ", duration=" + duration +
                ", age=" + age +
                '}';
    }
} 