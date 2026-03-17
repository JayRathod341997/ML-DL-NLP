# System Design Interview Questions - Linear Regression

## Table of Contents
1. [Architecture Diagrams](#architecture-diagrams)
2. [Follow-up Questions](#follow-up-questions)
3. [Short Q&A](#short-qa)

---

## Architecture Diagrams

### 1. ML Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ML PIPELINE ARCHITECTURE                            │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Raw Data  │────▶│   Data      │────▶│   Feature  │────▶│   Model     │
│   Sources   │     │   Loading   │     │   Engineer │     │   Training  │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                           │                    │                   │
                           ▼                    ▼                   ▼
                    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
                    │  Validation │     │  Transform  │     │  Evaluation │
                    │  & Cleaning │     │  & Scaling  │     │  & Testing  │
                    └─────────────┘     └─────────────┘     └─────────────┘
                                                                      │
                                                                      ▼
                                                               ┌─────────────┐
                                                               │   Model     │
                                                               │   Registry  │
                                                               └─────────────┘
                                                                      │
                    ┌─────────────────────────────────────────────────────┘
                    ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   API       │────▶│   Model     │────▶│   Business │────▶│   Feedback  │
│   Gateway   │     │   Serving   │     │   Logic    │     │   Loop      │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

### 2. Production ML System Design

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PRODUCTION ML SYSTEM DESIGN                              │
└─────────────────────────────────────────────────────────────────────────────┘

                                    ┌──────────────────┐
                                    │   API Gateway    │
                                    │   (REST/gRPC)    │
                                    └────────┬─────────┘
                                             │
                     ┌───────────────────────┼───────────────────────┐
                     │                       │                       │
                     ▼                       ▼                       ▼
            ┌───────────────┐      ┌───────────────┐      ┌───────────────┐
            │  Load         │      │  Auth &       │      │  Request      │
            │  Balancer     │      │  Validation   │      │  Logging      │
            └───────────────┘      └───────────────┘      └───────────────┘
                     │                       │                       │
                     └───────────────────────┼───────────────────────┘
                                             │
                                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MODEL SERVING LAYER                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     ┌─────────────────┐                              │    │
│  │                     │   Model Server  │                              │    │
│  │                     │   (Container)   │                              │    │
│  │                     └────────┬────────┘                              │    │
│  │                              │                                        │    │
│  │        ┌─────────────────────┼─────────────────────┐                 │    │
│  │        │                     │                     │                 │    │
│  │        ▼                     ▼                     ▼                 │    │
│  │  ┌──────────┐         ┌──────────┐         ┌──────────┐             │    │
│  │  │  Model   │         │  Model   │         │  Model   │             │    │
│  │  │  v1.0    │         │  v1.1    │         │  v2.0    │             │    │
│  │  └──────────┘         └──────────┘         └──────────┘             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                             │
                                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA LAYER                                        │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   Feature  │    │   Model     │    │   Metrics  │    │   Logs     │  │
│  │   Store    │    │   Registry  │    │   Store    │    │   Store    │  │
│  │  (Redis)   │    │  (MLflow)   │    │ (Prometheus)│   │ (Elastic)  │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3. Continuous Training Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CONTINUOUS TRAINING (CT) PIPELINE                        │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   New Data  │────▶│   Data      │────▶│   Feature  │────▶│   Model     │
│   Arrives   │     │   Ingestion │     │   Store    │     │   Training  │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                                                                      │
                     ┌─────────────────────────────────────────────────┘
                     │
                     ▼
              ┌─────────────┐
              │  Model      │
              │  Validation│
              └─────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
┌───────────────┐          ┌───────────────┐
│ Validation    │          │ Validation    │
│ PASSED        │          │ FAILED        │
└───────┬───────┘          └───────┬───────┘
        │                          │
        ▼                          ▼
┌───────────────┐          ┌───────────────┐
│  Promote to   │          │  Alert &      │
│  Production   │          │  Rollback     │
└───────────────┘          └───────────────┘
```

---

## Follow-up Questions

### 1. How would you handle data drift in production?

**Answer:**
```
Implementation Strategy:

1. MONITORING
   - Track feature distributions continuously
   - Monitor prediction distributions
   - Set up alerts for anomalies

2. DETECTION METHODS
   - Population Stability Index (PSI)
   - Kolmogorov-Smirnov test
   - Chi-squared test for categorical features

3. RESPONSE
   - Automated retraining pipeline
   - A/B testing new models
   - Gradual rollout

4. PREVENTION
   - Feature engineering with stable features
   - Regular model retraining schedule
   - Model ensembling
```

---

### 2. How would you design a model serving system for 1M requests/day?

**Answer:**
```
High-Level Design:

1. API GATEWAY
   - Rate limiting
   - Authentication
   - Request routing
   - Load balancing

2. MODEL SERVER
   - Horizontal scaling with Kubernetes
   - Multiple model versions
   - Batch processing support
   - GPU/CPU optimization

3. CACHING LAYER
   - Redis for feature caching
   - Prediction caching for similar inputs
   - TTL management

4. MONITORING
   - Latency percentiles (p50, p95, p99)
   - Error rates
   - Resource utilization

5. SCALING
   - Auto-scaling based on request volume
   - Queue-based processing for batch jobs
   - Geographic distribution
```

---

### 3. How do you ensure model reliability?

**Answer:**
```
Reliability Checklist:

1. INPUT VALIDATION
   - Schema validation
   - Range checking
   - Missing value handling

2. OUTPUT VALIDATION
   - Range checks (prices >= 0)
   - Anomaly detection
   - Fallback predictions

3. ERROR HANDLING
   - Circuit breakers
   - Retry logic
   - Graceful degradation

4. MONITORING
   - Health checks
   - Performance metrics
   - Data quality alerts

5. VERSIONING
   - Model versioning
   - Feature store versioning
   - Configuration versioning
```

---

### 4. How would you implement CI/CD for ML?

**Answer:**
```
ML CI/CD Pipeline:

1. CONTINUOUS INTEGRATION
   - Code linting & testing
   - Data validation tests
   - Model unit tests
   - Feature engineering tests

2. CONTINUOUS DELIVERY
   - Automated model training
   - Model evaluation
   - Model registry update
   - Staging deployment

3. CONTINUOUS TRAINING
   - Trigger on new data
   - Automated retraining
   - A/B testing
   - Rollback capabilities

4. TOOLS
   - MLflow for tracking
   - Kubeflow for orchestration
   - Prometheus/Grafana for monitoring
   - GitHub Actions for CI/CD
```

---

## Short Q&A

### Fundamentals

| Question | Answer |
|----------|--------|
| **What is Linear Regression?** | A supervised learning algorithm that models the relationship between variables by fitting a linear equation to observed data. |
| **What are the assumptions of Linear Regression?** | 1) Linearity, 2) Independence, 3) Homoscedasticity, 4) Normality of residuals, 5) No multicollinearity |
| **How do you check if Linear Regression is appropriate?** | Check scatter plot for linear relationship, calculate correlation coefficient, verify residual plots |
| **What is R² score?** | Coefficient of determination - measures the proportion of variance explained by the model (0-1, higher is better) |
| **What is RMSE?** | Root Mean Squared Error - square root of average squared differences between predictions and actual values |
| **What is the difference between Ridge and Lasso?** | Ridge uses L2 regularization (shrinks coefficients), Lasso uses L1 regularization (can set coefficients to zero - feature selection) |
| **What is multicollinearity?** | High correlation between independent variables, causing unstable coefficient estimates |
| **How do you detect multicollinearity?** | Variance Inflation Factor (VIF) - VIF > 10 indicates problematic multicollinearity |
| **What is regularization?** | Technique to prevent overfitting by adding a penalty term to the loss function |
| **What is the bias-variance tradeoff?** | High bias = underfitting, high variance = overfitting. Goal is to find balance |

---

### System Design

| Question | Answer |
|----------|--------|
| **How would you scale prediction serving?** | Use load balancers, horizontal scaling, caching, batch processing, async processing |
| **How do you handle missing values in production?** | Pre-fitted imputers, default values, validation checks, logging and alerting |
| **What is a feature store?** | Centralized repository for storing and serving ML features, ensures consistency between training and serving |
| **How would you implement model monitoring?** | Track prediction distributions, data drift, model performance metrics, setup alerts |
| **What is A/B testing for models?** | Testing new model on subset of traffic to compare performance before full rollout |
| **How do you version ML models?** | Use model registry, semantic versioning, track parameters and metrics with MLflow |
| **What is the difference between online and batch prediction?** | Online = real-time, low latency; Batch = processed in groups, higher throughput |
| **How would you handle model drift?** | Monitor data and model metrics, automated retraining, rollback mechanisms |

---

### Production Issues

| Question | Answer |
|----------|--------|
| **Why would model accuracy drop in production?** | Data drift, concept drift, upstream data changes, feature pipeline bugs |
| **How do you debug a model performing poorly?** | Check data pipeline, verify feature engineering, analyze feature importance, check for data leakage |
| **What causes prediction latency?** | Model size, feature computation, database queries, network latency, lack of caching |
| **How do you handle prediction failures?** | Circuit breakers, fallback predictions, retry logic, proper error logging |
| **What is data leakage?** | Using information from outside training data, leads to overly optimistic metrics |

---

### Interview Tips

| Topic | Key Points |
|-------|------------|
| **Explain Linear Regression** | Start with simple definition, explain cost function (MSE), mention gradient descent, discuss interpretation |
| **Discuss assumptions** | Be thorough - linearity, independence, homoscedasticity, normality, no multicollinearity |
| **Regularization explanation** | Explain L1 vs L2, when to use each, effect on coefficients |
| **Model evaluation** | Know R², RMSE, MAE, when to use each, overfitting vs underfitting |
| **Production readiness** | Mention monitoring, logging, error handling, versioning, scalability |

---

### Code Examples

**1. Simple Linear Regression**
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

**2. Ridge Regression with Cross-Validation**
```python
from sklearn.linear_model import RidgeCV
model = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5)
model.fit(X_train, y_train)
print(f"Best alpha: {model.alpha_}")
```

**3. Pipeline with Preprocessing**
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Ridge())
])
pipeline.fit(X_train, y_train)
```

---

## Additional Resources

- [Scikit-learn Linear Regression Documentation](https://scikit-learn.org/stable/modules/linear_model.html)
- [Google ML Engineering Best Practices](https://developers.google.com/machine-learning/engineering)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
