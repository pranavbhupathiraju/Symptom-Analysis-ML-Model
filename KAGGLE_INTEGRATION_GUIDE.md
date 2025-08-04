# 🎯 Kaggle Dataset Integration & ML Fine-tuning Guide

## 📋 **Prerequisites**

### 1. Kaggle API Setup
```bash
# Install Kaggle CLI
pip install kaggle

# Get your API credentials from Kaggle:
# 1. Go to kaggle.com → Account → Create API Token
# 2. Download kaggle.json
# 3. Place it in ~/.kaggle/kaggle.json
mkdir -p ~/.kaggle
# Copy your kaggle.json to ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 2. Test Kaggle Connection
```bash
kaggle datasets list --limit 5
```

## 🔍 **Finding Medical Datasets**

### Popular Medical Datasets on Kaggle:
1. **Disease Symptom Prediction**: `itachi9604/disease-symptom-description-dataset`
2. **Medical Symptoms**: `priyanshusethi/medical-symptoms-dataset`
3. **Health Care Analytics**: `fedesoriano/stroke-prediction-dataset`
4. **COVID-19 Symptoms**: `covid19-symptoms-dataset`

### Search Command:
```bash
kaggle datasets list --search "medical symptoms" --limit 10
```

## 📥 **Downloading & Preprocessing**

### Step 1: Download Dataset
```bash
# Example: Download disease symptoms dataset
kaggle datasets download -d itachi9604/disease-symptom-description-dataset
unzip disease-symptom-description-dataset.zip
```

### Step 2: Data Exploration Script
```python
# explore_dataset.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def explore_medical_dataset(file_path):
    """Explore and understand the medical dataset"""
    
    # Load data
    df = pd.read_csv(file_path)
    print("Dataset Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Check for missing values
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # Data types
    print("\nData types:")
    print(df.dtypes)
    
    # Basic statistics
    print("\nBasic statistics:")
    print(df.describe())
    
    return df

if __name__ == "__main__":
    df = explore_medical_dataset("your_dataset.csv")
```

## 🔧 **Data Preprocessing & Feature Engineering**

### Step 3: Preprocessing Script
```python
# preprocess_medical_data.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
import re

def clean_symptoms(symptoms_str):
    """Clean and standardize symptoms"""
    if pd.isna(symptoms_str):
        return []
    
    # Convert to lowercase and split
    symptoms = str(symptoms_str).lower().split(',')
    
    # Clean each symptom
    cleaned = []
    for symptom in symptoms:
        symptom = re.sub(r'[^\w\s]', '', symptom.strip())
        if symptom and len(symptom) > 2:
            cleaned.append(symptom)
    
    return cleaned

def preprocess_medical_data(df):
    """Preprocess medical dataset for ML training"""
    
    # Clean symptoms
    df['symptoms_cleaned'] = df['symptoms'].apply(clean_symptoms)
    
    # Create age groups if age exists
    if 'age' in df.columns:
        df['age_group'] = pd.cut(df['age'], 
                                bins=[0, 18, 35, 50, 65, 100], 
                                labels=['child', 'young_adult', 'adult', 'senior', 'elderly'])
    
    # Create duration groups if duration exists
    if 'duration' in df.columns:
        df['duration_group'] = pd.cut(df['duration'], 
                                    bins=[0, 1, 3, 7, 30, 365], 
                                    labels=['acute', 'subacute', 'week', 'month', 'chronic'])
    
    return df

def prepare_features_for_ml(df):
    """Prepare features for machine learning"""
    
    # One-hot encode symptoms
    mlb = MultiLabelBinarizer()
    symptoms_encoded = mlb.fit_transform(df['symptoms_cleaned'])
    symptoms_df = pd.DataFrame(symptoms_encoded, columns=mlb.classes_)
    
    # Combine with other features
    feature_columns = []
    
    if 'age' in df.columns:
        feature_columns.append('age')
    if 'duration' in df.columns:
        feature_columns.append('duration')
    if 'age_group' in df.columns:
        # Encode age groups
        le_age = LabelEncoder()
        age_encoded = le_age.fit_transform(df['age_group'])
        df['age_group_encoded'] = age_encoded
        feature_columns.append('age_group_encoded')
    
    # Combine all features
    X = pd.concat([symptoms_df, df[feature_columns]], axis=1)
    
    return X, mlb, feature_columns

if __name__ == "__main__":
    # Load your dataset
    df = pd.read_csv("your_dataset.csv")
    
    # Preprocess
    df_processed = preprocess_medical_data(df)
    
    # Prepare features
    X, mlb, feature_cols = prepare_features_for_ml(df_processed)
    
    print("Feature matrix shape:", X.shape)
    print("Number of symptoms:", len(mlb.classes_))
```

## 🤖 **Model Training & Fine-tuning**

### Step 4: Advanced Model Training
```python
# train_advanced_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def train_advanced_model(X, y, model_type='random_forest'):
    """Train advanced model with hyperparameter tuning"""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'random_forest': RandomForestClassifier(random_state=42),
        'gradient_boosting': GradientBoostingClassifier(random_state=42),
        'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
        'svm': SVC(random_state=42, probability=True)
    }
    
    # Hyperparameter grids
    param_grids = {
        'random_forest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10]
        },
        'gradient_boosting': {
            'n_estimators': [100, 200],
            'learning_rate': [0.1, 0.05],
            'max_depth': [3, 5, 7]
        },
        'logistic_regression': {
            'C': [0.1, 1, 10],
            'penalty': ['l1', 'l2']
        },
        'svm': {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear']
        }
    }
    
    # Train with GridSearch
    model = models[model_type]
    param_grid = param_grids[model_type]
    
    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring='accuracy', n_jobs=-1
    )
    grid_search.fit(X_train_scaled, y_train)
    
    # Best model
    best_model = grid_search.best_estimator_
    
    # Evaluate
    y_pred = best_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Cross-validation score: {grid_search.best_score_:.3f}")
    print(f"Test accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return best_model, scaler, accuracy

def save_model_and_artifacts(model, scaler, mlb, feature_columns, model_name):
    """Save model and preprocessing artifacts"""
    
    joblib.dump(model, f'{model_name}_model.pkl')
    joblib.dump(scaler, f'{model_name}_scaler.pkl')
    joblib.dump(mlb, f'{model_name}_mlb.pkl')
    joblib.dump(feature_columns, f'{model_name}_features.pkl')
    
    print(f"Model and artifacts saved as {model_name}_*.pkl")

if __name__ == "__main__":
    # Load your preprocessed data
    # X, y = your_data_loading_function()
    
    # Train multiple models
    models_results = {}
    
    for model_type in ['random_forest', 'gradient_boosting', 'logistic_regression']:
        print(f"\n{'='*50}")
        print(f"Training {model_type.upper()}")
        print(f"{'='*50}")
        
        model, scaler, accuracy = train_advanced_model(X, y, model_type)
        models_results[model_type] = {
            'model': model,
            'scaler': scaler,
            'accuracy': accuracy
        }
    
    # Find best model
    best_model_type = max(models_results.keys(), 
                         key=lambda x: models_results[x]['accuracy'])
    
    print(f"\nBest model: {best_model_type} with accuracy: {models_results[best_model_type]['accuracy']:.3f}")
    
    # Save best model
    save_model_and_artifacts(
        models_results[best_model_type]['model'],
        models_results[best_model_type]['scaler'],
        mlb,  # from your preprocessing
        feature_columns,  # from your preprocessing
        f'kaggle_{best_model_type}'
    )
```

## 🔄 **Update Flask Service**

### Step 5: Enhanced Flask Service
```python
# app_kaggle_model.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import logging

app = Flask(__name__)
CORS(app)

# Load Kaggle-trained model
try:
    model = joblib.load('kaggle_random_forest_model.pkl')
    scaler = joblib.load('kaggle_random_forest_scaler.pkl')
    mlb = joblib.load('kaggle_random_forest_mlb.pkl')
    feature_columns = joblib.load('kaggle_random_forest_features.pkl')
    print("Kaggle model loaded successfully!")
except Exception as e:
    print(f"Error loading Kaggle model: {e}")
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        # Extract features
        symptoms = data.get('symptoms', [])
        duration = data.get('duration', 0)
        age = data.get('age', 30)
        
        # Prepare features
        symptoms_encoded = mlb.transform([symptoms])
        features = np.zeros((1, len(feature_columns)))
        
        # Add encoded symptoms
        features[0, :len(mlb.classes_)] = symptoms_encoded[0]
        
        # Add other features
        if 'age' in feature_columns:
            age_idx = feature_columns.index('age')
            features[0, age_idx] = age
        
        if 'duration' in feature_columns:
            duration_idx = feature_columns.index('duration')
            features[0, duration_idx] = duration
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Predict
        prediction = model.predict(features_scaled)[0]
        confidence = np.max(model.predict_proba(features_scaled))
        
        # Map to actions
        action_mapping = {
            'low': 'Monitor at home',
            'moderate': 'Visit urgent care', 
            'high': 'Go to ER immediately'
        }
        
        return jsonify({
            'risk_level': prediction,
            'suggested_action': action_mapping.get(prediction, 'Consult doctor'),
            'confidence_score': float(confidence),
            'model_type': 'kaggle_dataset_trained'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5002)
```

## 📊 **Model Evaluation & Comparison**

### Step 6: Model Comparison Script
```python
# compare_models.py
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def compare_models(synthetic_model, kaggle_model, test_data):
    """Compare synthetic vs Kaggle-trained models"""
    
    # Test both models
    synthetic_predictions = synthetic_model.predict(test_data)
    kaggle_predictions = kaggle_model.predict(test_data)
    
    # Compare accuracy
    synthetic_accuracy = accuracy_score(y_test, synthetic_predictions)
    kaggle_accuracy = accuracy_score(y_test, kaggle_predictions)
    
    print(f"Synthetic Model Accuracy: {synthetic_accuracy:.3f}")
    print(f"Kaggle Model Accuracy: {kaggle_accuracy:.3f}")
    
    # Detailed comparison
    print("\nSynthetic Model Report:")
    print(classification_report(y_test, synthetic_predictions))
    
    print("\nKaggle Model Report:")
    print(classification_report(y_test, kaggle_predictions))
    
    return synthetic_accuracy, kaggle_accuracy
```

## 🚀 **Complete Workflow**

### Step 7: Full Integration Script
```python
# integrate_kaggle_dataset.py
import os
import subprocess
import pandas as pd
from preprocess_medical_data import preprocess_medical_data, prepare_features_for_ml
from train_advanced_model import train_advanced_model, save_model_and_artifacts

def integrate_kaggle_dataset(dataset_name, target_column):
    """Complete workflow to integrate Kaggle dataset"""
    
    print(f"Integrating Kaggle dataset: {dataset_name}")
    
    # 1. Download dataset
    print("Downloading dataset...")
    subprocess.run(['kaggle', 'datasets', 'download', '-d', dataset_name])
    
    # 2. Extract and explore
    print("Extoring dataset...")
    subprocess.run(['unzip', f'{dataset_name.split("/")[-1]}.zip'])
    
    # 3. Load and preprocess
    print("Preprocessing data...")
    # Find the CSV file
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    if csv_files:
        df = pd.read_csv(csv_files[0])
        df_processed = preprocess_medical_data(df)
        
        # 4. Prepare features
        X, mlb, feature_cols = prepare_features_for_ml(df_processed)
        y = df_processed[target_column]
        
        # 5. Train model
        print("Training model...")
        model, scaler, accuracy = train_advanced_model(X, y, 'random_forest')
        
        # 6. Save model
        save_model_and_artifacts(model, scaler, mlb, feature_cols, 'kaggle_integrated')
        
        print(f"Integration complete! Model accuracy: {accuracy:.3f}")
        return True
    
    return False

if __name__ == "__main__":
    # Example usage
    integrate_kaggle_dataset('itachi9604/disease-symptom-description-dataset', 'disease')
```

## 📈 **Performance Monitoring**

### Step 8: Model Performance Tracking
```python
# performance_monitor.py
import time
import json
from datetime import datetime

class ModelPerformanceMonitor:
    def __init__(self, model_name):
        self.model_name = model_name
        self.predictions = []
        self.response_times = []
        
    def log_prediction(self, input_data, prediction, confidence, response_time):
        """Log prediction performance"""
        self.predictions.append({
            'timestamp': datetime.now().isoformat(),
            'input': input_data,
            'prediction': prediction,
            'confidence': confidence,
            'response_time': response_time
        })
        
    def get_performance_stats(self):
        """Get performance statistics"""
        if not self.predictions:
            return {}
        
        confidences = [p['confidence'] for p in self.predictions]
        response_times = [p['response_time'] for p in self.predictions]
        
        return {
            'total_predictions': len(self.predictions),
            'avg_confidence': np.mean(confidences),
            'avg_response_time': np.mean(response_times),
            'model_name': self.model_name
        }
    
    def save_performance_log(self, filename):
        """Save performance log to file"""
        with open(filename, 'w') as f:
            json.dump(self.predictions, f, indent=2)
```

## 🎯 **Key Learning Points**

1. **Data Quality**: Real datasets often have missing values, inconsistencies
2. **Feature Engineering**: Critical for medical data (age groups, duration categories)
3. **Model Selection**: Different algorithms perform differently on medical data
4. **Validation**: Cross-validation is crucial for medical applications
5. **Interpretability**: Medical models need to be explainable
6. **Performance Monitoring**: Track model drift and accuracy over time

## 🔧 **Troubleshooting Common Issues**

1. **Kaggle API Issues**: Check credentials and permissions
2. **Memory Issues**: Use data chunking for large datasets
3. **Class Imbalance**: Use SMOTE or class weights
4. **Feature Scaling**: Always scale features for ML algorithms
5. **Model Persistence**: Save all preprocessing artifacts

This guide gives you the complete workflow to integrate real Kaggle datasets and fine-tune your ML models for production use! 