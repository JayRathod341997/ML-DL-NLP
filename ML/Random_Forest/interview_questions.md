# Random Forest - Interview Questions & System Design

## Table of Contents
1. [Fundamentals](#fundamentals)
2. [System Design](#system-design)
3. [Production Issues](#production-issues)
4. [Short Q&A](#short-qa)

---

## Fundamentals

### What is Random Forest?

Random Forest is an ensemble learning method that builds multiple decision trees during training and outputs the class that is the mode of the classes (classification) or mean prediction (regression) of individual trees.

### How does Random Forest work?

1. **Bootstrap Sampling**: Randomly sample with replacement from training data
2. **Feature Selection**: Randomly select subset of features at each split
3. **Build Trees**: Grow decision trees on bootstrapped samples
4. **Ensemble Prediction**: Aggregate predictions from all trees

### Why is Random Forest better than a single decision tree?

- **Reduced Overfitting**: Averaging multiple trees reduces variance
- **Handles Missing Values**: Can handle missing data
- **Feature Importance**: Provides feature importance scores
- **Robust to Outliers**: Less sensitive to outliers

### What are the key hyperparameters?

| Parameter | Description |
|-----------|-------------|
| `n_estimators` | Number of trees |
| `max_depth` | Maximum tree depth |
| `min_samples_split` | Min samples to split a node |
| `min_samples_leaf` | Min samples in leaf node |
| `max_features` | Features to consider for split |

---

## System Design

### ML Pipeline Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Raw Data  │────▶│   Data      │────▶│   Feature  │────▶│   Model     │
│   Sources   │     │   Loading   │     │   Engineer │     │   Training  │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                                                                      │
                                    ┌─────────────────────────────────┘
                                    ▼
                            ┌─────────────┐
                            │   Ensemble  │
                            │   Learning  │
                            └─────────────┘
```

### Key Design Decisions

1. **Tree Count**: Start with 100 trees, increase if needed
2. **Parallel Processing**: Use `n_jobs=-1` for parallel training
3. **Feature Importance**: Use Gini importance or permutation importance

---

## Production Issues

### 1. Overfitting

**Problem**: Model performs well on training but poorly on test

**Solutions**:
- Reduce tree depth
- Increase min_samples_leaf
- Use cross-validation
- Prune trees

### 2. Class Imbalance

**Problem**: Uneven class distribution

**Solutions**:
- Use `class_weight='balanced'`
- SMOTE oversampling
- Adjust threshold

### 3. Large Model Size

**Problem**: Too many trees make deployment difficult

**Solutions**:
- Reduce n_estimators
- Use model compression
- Export only essential trees

---

## Short Q&A

| Question | Answer |
|----------|--------|
| **What is bagging?** | Bootstrap aggregating - combining predictions from multiple models trained on different bootstrap samples |
| **What is the difference between Random Forest and Gradient Boosting?** | Random Forest uses parallel tree building with bagging; Gradient Boosting uses sequential tree building with boosting |
| **How do you handle missing values in Random Forest?** | Can handle missing values natively; use imputation for better results |
| **What is Out-of-Bag (OOB) score?** | Error rate calculated on samples not used in training (bootstrap) |
| **Why use random feature selection at each split?** | Reduces correlation between trees, improving ensemble performance |
| **How does Random Forest handle categorical variables?** | Requires encoding; can handle natively in some implementations |
| **What is the time complexity?** | O(n * m * log(n)) where n = samples, m = features |
| **When should you NOT use Random Forest?** | When model interpretability is crucial, or when dealing with very sparse high-dimensional data |

---

## Follow-up Questions

### How would you optimize Random Forest for production?

```
1. Hyperparameter Tuning
   - Use RandomizedSearchCV or Optuna
   - Focus on: n_estimators, max_depth, min_samples_split

2. Feature Selection
   - Remove low-importance features
   - Reduce model complexity

3. Model Compression
   - Prune decision trees
   - Use only top N important trees

4. Caching
   - Cache predictions for similar inputs
   - Use feature hashing for new categories
```

### How would you deploy Random Forest for real-time predictions?

```
1. Model Serving
   - Use ONNX for cross-platform deployment
   - Containerize with Docker
   - Scale horizontally with Kubernetes

2. Optimization
   - Batch predictions for throughput
   - Precompute feature transformations
   - Use efficient data structures

3. Monitoring
   - Track prediction latency
   - Monitor feature distributions
   - Setup drift detection
```

### How do you interpret Random Forest predictions?

```
1. Feature Importance
   - Gini importance: mean decrease in impurity
   - Permutation importance: accuracy decrease when feature shuffled

2. SHAP Values
   - Explain individual predictions
   - Show feature contributions

3. Partial Dependence Plots
   - Show marginal effect of features
```
