# Production Issues & Solutions

This document outlines common production issues encountered when deploying Linear Regression models and their solutions.

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
import numpy as np

def detect_drift(reference_data, production_data, threshold=0.1):
    """
    Detect data drift using population stability index (PSI)
    """
    drift_detected = {}
    
    for col in reference_data.columns:
        ref_bins = np.percentile(reference_data[col], [0, 20, 40, 60, 80, 100])
        prod_counts = np.histogram(production_data[col], bins=ref_bins)[0]
        ref_counts = np.histogram(reference_data[col], bins=ref_bins)[0]
        
        prod_pct = prod_counts / len(production_data)
        ref_pct = ref_counts / len(reference_data)
        
        prod_pct = np.where(prod_pct == 0, 0.0001, prod_pct)
        ref_pct = np.where(ref_pct == 0, 0.0001, ref_pct)
        
        psi = np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct))
        
        if psi > threshold:
            drift_detected[col] = psi
    
    return drift_detected
```

**B. Automated Retraining Pipeline**
```python
if drift_detected:
    logger.warning(f"Data drift detected: {drift_detected}")
    trigger_retraining_pipeline()
```

---

## 2. Multicollinearity

### Issue
High correlation between independent variables causes unstable coefficient estimates.

### Symptoms
- Coefficients change dramatically with small data changes
- Large variance in coefficient estimates
- Model becomes unreliable

### Solutions

**A. Use Ridge Regression**
```python
from sklearn.linear_model import Ridge

# Ridge regression with L2 regularization
model = Ridge(alpha=1.0)  # Higher alpha = more regularization
```

**B. Calculate VIF (Variance Inflation Factor)**
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    return vif_data
```

---

## 3. Overfitting

### Issue
The model performs well on training data but poorly on new data.

### Solutions

**A. Regularization**
```python
# Ridge (L2) - shrinks coefficients
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1.0)

# Lasso (L1) - feature selection
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.1)

# ElasticNet - combination
from sklearn.linear_model import ElasticNet
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
```

**B. Cross-Validation**
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f"CV R² scores: {scores}, Mean: {scores.mean():.3f}")
```

---

## 4. Feature Scaling Issues

### Issue
Features with different scales affect model performance.

### Solutions

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# For prediction
X_new_scaled = scaler.transform(X_new)
```

---

## 5. Missing Values in Production

### Issue
Missing values in production data cause prediction failures.

### Solutions

```python
def handle_missing_production(X, imputer):
    """Handle missing values in production data."""
    # Use pre-fitted imputer
    X_imputed = imputer.transform(X)
    
    # Check for any remaining NaN
    if np.isnan(X_imputed).any():
        logger.warning("Remaining NaN values after imputation")
        X_imputed = np.nan_to_num(X_imputed, nan=0)
    
    return X_imputed
```

---

## 6. Model Versioning Issues

### Issue
Confusion between different model versions in production.

### Solutions

```python
import mlflow

# Track model version
mlflow.log_param("model_type", "Ridge")
mlflow.log_param("alpha", 1.0)
mlflow.log_metric("r2_score", 0.85)

# Save model with version
model_version = "v1.0.0"
model.save(f"models/house_price_{model_version}.joblib")
```

---

## 7. Debugging Techniques

### A. Check Data Pipeline
```python
# Verify data shapes at each step
logger.info(f"Raw data shape: {df.shape}")
logger.info(f"After preprocessing: {X.shape}")
logger.info(f"Train/Test split: {X_train.shape}/{X_test.shape}")
```

### B. Validate Model Inputs
```python
def validate_input(X, expected_features):
    """Validate input data before prediction."""
    if X.shape[1] != len(expected_features):
        raise ValueError(f"Expected {len(expected_features)} features, got {X.shape[1]}")
    
    if np.isnan(X).any():
        logger.warning("NaN values found in input")
    
    return True
```

### C. Monitor Prediction Distribution
```python
def check_prediction_stats(y_pred):
    """Monitor prediction statistics."""
    logger.info(f"Predictions - Min: {y_pred.min():.2f}, Max: {y_pred.max():.2f}, Mean: {y_pred.mean():.2f}")
    
    if y_pred.min() < 0:
        logger.warning("Negative predictions found!")
    
    if y_pred.std() < 100:
        logger.warning("Very low variance in predictions!")
```

---

## 8. Performance Optimization

### Issue
Slow prediction times in production.

### Solutions

**A. Batch Predictions**
```python
def predict_batch(model, X_batch, batch_size=1000):
    """Process predictions in batches."""
    predictions = []
    for i in range(0, len(X_batch), batch_size):
        batch = X_batch[i:i+batch_size]
        predictions.extend(model.predict(batch))
    return np.array(predictions)
```

**B. Model Optimization**
```python
# Use joblib for faster loading
import joblib
model = joblib.load('model.joblib')

# Use n_jobs for parallel processing
model = Ridge(alpha=1.0, n_jobs=-1)
```

---

## 9. Logging Best Practices

### A. Structured Logging
```python
import json

logger.info(json.dumps({
    "event": "prediction",
    "model_version": "1.0.0",
    "input_shape": X.shape,
    "prediction_mean": float(y_pred.mean()),
    "timestamp": datetime.now().isoformat()
}))
```

### B. Log Rotations
```python
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler(
    'app.log',
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
```

---

## 10. Error Handling

### A. Graceful Degradation
```python
try:
    prediction = model.predict(X)
except Exception as e:
    logger.error(f"Prediction failed: {e}")
    # Return fallback prediction
    return default_prediction
```

### B. Circuit Breaker Pattern
```python
class ModelCircuitBreaker:
    def __init__(self, failure_threshold=5):
        self.failures = 0
        self.failure_threshold = failure_threshold
        self.is_open = False
    
    def call(self, func):
        if self.is_open:
            raise Exception("Circuit breaker is open")
        
        try:
            result = func()
            self.failures = 0
            return result
        except Exception as e:
            self.failures += 1
            if self.failures >= self.failure_threshold:
                self.is_open = True
            raise
```
