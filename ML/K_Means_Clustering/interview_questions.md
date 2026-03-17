# K-Means Clustering - Interview Questions & System Design

## Fundamentals

### What is K-Means?

Unsupervised learning algorithm that partitions data into K clusters by minimizing within-cluster variance.

### How does K-Means work?

1. Initialize K centroids (random or K-means++)
2. Assign each point to nearest centroid
3. Recalculate centroids
4. Repeat until convergence

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `n_clusters` | Number of clusters (K) |
| `max_iter` | Maximum iterations |
| `n_init` | Number of initializations |
| `init` | Initialization method |
| `algorithm` | 'lloyd', 'elkan', 'full' |

---

## System Design

### Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Data      │────▶│   Distance  │────▶│   Cluster   │
│   Input     │     │   Compute   │     │   Assignment│
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                    ┌──────────────────────────┘
                    ▼
            ┌─────────────┐
            │   Centroid  │
            │   Update    │
            └─────────────┘
```

---

## Production Issues

### 1. Choosing K

**Problem**: Unknown optimal number of clusters

**Solutions**:
- Elbow method
- Silhouette score
- Gap statistic

### 2. Initialization

**Problem**: Poor initialization leads to local optima

**Solutions**:
- Use K-means++ initialization
- Run multiple times (n_init)
- Increase n_init

### 3. Scalability

**Problem**: Slow for large datasets

**Solutions**:
- Mini-batch K-means
- Use approximate methods
- Dimensionality reduction first

---

## Short Q&A

| Question | Answer |
|----------|--------|
| **What is inertia?** | Sum of squared distances to closest centroid |
| **What is K-means++?** | Initialization method that improves convergence |
| **When does K-means fail?** | Non-spherical clusters, varying densities, high dimensions |
| **What is mini-batch K-means?** | Uses random mini-batches instead of full dataset |
| **Is K-means deterministic?** | No - depends on initialization |
| **How do you evaluate K-means?** | Silhouette score, elbow method, domain knowledge |

---

## Follow-up Questions

### How would you determine optimal K?

```
1. Elbow Method
   - Plot inertia vs K
   - Look for "elbow" point

2. Silhouette Analysis
   - Calculate silhouette score for different K
   - Higher score = better separation

3. Domain Knowledge
   - Business requirements
   - Interpretability needs
```
