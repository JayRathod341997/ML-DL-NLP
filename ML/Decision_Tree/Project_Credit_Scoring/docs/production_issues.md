# Production Issues & Solutions

This document outlines common production issues encountered when deploying the Credit Scoring Decision Tree model and their solutions.

---

## 1. Data Drift Detection

### Issue
The model's prediction accuracy degrades over time due to changes in data distribution (data drift).

### Symptoms
- Accuracy drops in production
- Feature distributions change significantly
- Model predictions seem less reliable

### Solutions

**A. Monitor Feature Statistics**
```python
# Track feature statistics in production
import numpy as np

def detect_drift(reference_data, production_data, threshold=0.1):
    """
    Detect data drift using population stability index (PSI)
    """
    drift_detected = {}
    
    for col in reference_data.columns:
        # Calculate PSI
        ref_bins = np.percentile(reference_data[col], [0, 20, 40, 60, 80, 100])
        prod_counts = np.histogram(production_data[col], bins=ref_bins)[0]
        ref_counts = np.histogram(reference_data[col], bins=ref_bins)[0]
        
        # Calculate percentages
        prod_pct = prod_counts / len(production_data)
        ref_pct = ref_counts / len(reference_data)
        
        # Add small value to avoid log(0)
        prod_pct = np.where(prod_pct == 0, 0.0001, prod_pct)
        ref_pct = np.where(ref_pct == 0, 0.0001, ref_pct)
        
        psi = np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct))
        
        if psi > threshold:
            drift_detected[col] = psi
    
    return drift_detected
```

**B. Automated Retraining Pipeline**
```python
# Trigger retraining when drift is detected
if drift_detected:
    logger.warning(f"Data drift detected: {drift_detected}")
    # Trigger model retraining
    trigger_retraining_pipeline()
```

---

## 2. Model Overfitting

### Issue
The model performs well on training data but poorly on new data.

### Solutions

**A. Pruning the Decision Tree**
```python
# Use max_depth and min_samples parameters
model = DecisionTreeClassifier(
    max_depth=5,              # Limit tree depth
    min_samples_split=10,    # Min samples to split
    min_samples_leaf=5,      # Min samples in leaf
    ccp_alpha=0.01           # Cost-complexity pruning
)
```

**B. Cross-Validation**
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5)
print(f"CV Scores: {scores}")
print(f"Mean: {scores.mean():.4f}")
```

**C. Regularization**
```python
# Use GridSearchCV for hyperparameter tuning
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [5, 10, 20],
    'min_samples_leaf': [2, 4, 6]
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(),
    param_grid,
    cv=5,
    scoring='accuracy'
)
```

---

## 3. API Latency Issues

### Issue
Predictions take too long, causing timeouts.

### Solutions

**A. Model Optimization**
```python
# Use simpler model for production
# Avoid unnecessary complexity

# Serialize model efficiently
import joblib
joblib.dump(model, 'model_compressed.joblib', compress=3)
```

**B. Caching**
```python
from functools import lru_cache
import hashlib
import json

@lru_cache(maxsize=1000)
def predict_cached(input_data):
    # Cache frequent predictions
    return model.predict(input_data)
```

**C. Batch Prediction**
```python
# Process multiple predictions at once
def batch_predict(data_list, batch_size=100):
    results = []
    for i in range(0, len(data_list), batch_size):
        batch = data_list[i:i+batch_size]
        results.extend(model.predict(batch))
    return results
```

---

## 4. Missing Data Handling

### Issue
Production data contains missing values that cause errors.

### Solutions

**A. Default Values**
```python
def preprocess_input(data):
    # Define default values for each feature
    defaults = {
        'age': 30,
        'income': 50000,
        'credit_score': 600,
        'employment_years': 5,
        'loan_amount': 10000
    }
    
    # Fill missing values
    for key, value in defaults.items():
        if key not in data or data[key] is None:
            data[key] = value
    
    return data
```

**B. Skip Invalid Records**
```python
def validate_input(data):
    required_fields = ['age', 'income', 'credit_score', 'employment_years', 'loan_amount']
    
    for field in required_fields:
        if field not in data or data[field] is None:
            raise ValueError(f"Missing required field: {field}")
    
    return True
```

---

## 5. Logging Best Practices

### Issue
Insufficient logging makes debugging difficult.

### Solutions

**A. Structured Logging**
```python
import json
import logging

class StructuredLogger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)
    
    def log_prediction(self, features, prediction, latency_ms):
        log_data = {
            "event": "prediction",
            "features": features,
            "prediction": int(prediction),
            "latency_ms": latency_ms
        }
        self.logger.info(json.dumps(log_data))
    
    def log_error(self, error, context):
        log_data = {
            "event": "error",
            "error": str(error),
            "context": context
        }
        self.logger.error(json.dumps(log_data))
```

**B. Request Tracing**
```python
import uuid
from functools import wraps

def trace_requests(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        request_id = str(uuid.uuid4())
        logger.info(f"Request {request_id} started")
        
        try:
            result = func(*args, **kwargs)
            logger.info(f"Request {request_id} completed")
            return result
        except Exception as e:
            logger.error(f"Request {request_id} failed: {str(e)}")
            raise
    
    return wrapper
```

---

## 6. Version Control for Models

### Issue
Difficulty tracking which model version is in production.

### Solutions

**A. Model Metadata**
```python
import json
from datetime import datetime

model_version = {
    "version": "1.0.0",
    "created_at": datetime.now().isoformat(),
    "algorithm": "DecisionTreeClassifier",
    "hyperparameters": {
        "max_depth": 5,
        "min_samples_split": 10,
        "min_samples_leaf": 5
    },
    "metrics": {
        "accuracy": 0.85,
        "precision": 0.83,
        "recall": 0.87
    },
    "training_data_version": "v2.1"
}

with open('model_metadata.json', 'w') as f:
    json.dump(model_version, f, indent=2)
```

**B. Model Registry**
```python
# Use MLflow for model versioning
import mlflow

mlflow.set_experiment("credit_scoring")

with mlflow.start_run():
    mlflow.log_param("max_depth", 5)
    mlflow.log_metric("accuracy", 0.85)
    mlflow.sklearn.log_model(model, "model")
```

---

## 7. Security Concerns

### Issue
Sensitive data exposure or model manipulation.

### Solutions

**A. Input Validation**
```python
def validate_features(data):
    """Validate input features"""
    age = data.get('age', 0)
    if age < 18 or age > 100:
        raise ValueError("Invalid age")
    
    income = data.get('income', 0)
    if income < 0:
        raise ValueError("Income cannot be negative")
    
    # ... more validations
```

**B. Rate Limiting**
```python
from flask_limiter import Limiter

limiter = Limiter(app, key_func=get_remote_address)

@app.route('/predict', methods=['POST'])
@limiter.limit("100 per minute")
def predict():
    # Prediction logic
    pass
```

**C. Audit Logging**
```python
import audit

# Log all access
audit.log("Model accessed by user: %s", user_id)
```

---

## 8. Testing in Production

### Issue
Need to validate model behavior before full deployment.

### Solutions

**A. Canary Deployment**
```bash
# Route 10% of traffic to new model
az ml online-deployment update \
  --name credit-scoring-v2 \
  --traffic-allocation 10
```

**B. A/B Testing**
```python
# Randomly assign model versions
import random

def get_model_for_user(user_id):
    if random.random() < 0.1:  # 10% get new model
        return new_model
    return current_model