# K-Nearest Neighbors - Interview Questions & System Design

## Fundamentals

### What is KNN?

K-Nearest Neighbors is a simple, instance-based learning algorithm that classifies new data points based on similarity to training data.

### How does KNN work?

1. Choose K (number of neighbors)
2. Calculate distance to all training points
3. Find K nearest neighbors
4. Vote/average their labels

### Key Hyperparameters

| Parameter | Description |
|-----------|-------------|
| `n_neighbors` (K) | Number of neighbors |
| `weights` | 'uniform' or 'distance' |
| `metric` | Distance metric (euclidean, manhattan, minkowski) |
| `p` | Power parameter for Minkowski |

### Distance Metrics

- **Euclidean**: √(Σ(xi-yi)²)
- **Manhattan**: Σ|xi-yi|
- **Minkowski**: (Σ|xi-yi|^p)^(1/p)

---

## System Design

### Production Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Request   │────▶│   Feature   │────▶│   KNN      │
│   API       │     │   Compute   │     │   Search   │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                    ┌──────────────────────────┘
                    ▼
            ┌─────────────┐
            │   Result    │
            │   Voting    │
            └─────────────┘
```

---

## Production Issues

### 1. Slow Prediction

**Problem**: O(n) time complexity for each prediction

**Solutions**:
- Use KD-Tree or Ball-Tree
- Dimensionality reduction
- Approximate nearest neighbors (FAISS)
- Feature hashing

### 2. Curse of Dimensionality

**Problem**: Distance metrics become meaningless in high dimensions

**Solutions**:
- PCA for dimensionality reduction
- Feature selection
- Use appropriate K

### 3. Feature Scaling

**Problem**: Features with large ranges dominate

**Solutions**:
- StandardScaler
- MinMaxScaler

---

## Short Q&A

| Question | Answer |
|----------|--------|
| **What is the best K value?** | Use cross-validation; odd K for binary classification |
| **Why use distance-weighted voting?** | Closer neighbors have more influence |
| **When is KNN not suitable?** | Large datasets, high dimensions, sparse data |
| **What is the difference between KNN classification and regression?** | Classification uses majority vote; regression uses mean/average |
| **How do you handle categorical features?** | Use Hamming distance or one-hot encode |
| **What is the time complexity?** | O(n * d) for brute force, O(n log n) with KD-Tree |
| **Is KNN parametric or non-parametric?** | Non-parametric - no assumptions about data distribution |

---

## Follow-up Questions

### How would you optimize KNN for large datasets?

```
1. Algorithmic Optimizations
   - Use Ball-Tree or KD-Tree
   - Approximate nearest neighbors (ANN)
   - Use FAISS library

2. Data Optimizations
   - Reduce dimensionality with PCA
   - Feature selection
   - Data subsampling

3. System Optimizations
   - In-memory data with Redis
   - Batch prediction
   - Precompute distances
```

### How do you choose K value?

```
1. Cross-validation
   - Try K from 1 to sqrt(n)
   - Select K with best CV score

2. Elbow Method
   - Plot error vs K
   - Choose K at the "elbow"

3. Consider Class Balance
   - Use odd K for binary classification
   - Consider class weights
```
