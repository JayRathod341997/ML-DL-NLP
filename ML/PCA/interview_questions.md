# PCA (Principal Component Analysis) - Interview Questions & System Design

## Fundamentals

### What is PCA?

Dimensionality reduction technique that transforms data to a new coordinate system of principal components.

### How does PCA work?

1. Center data (subtract mean)
2. Compute covariance matrix
3. Find eigenvalues and eigenvectors
4. Project data onto top k components

### Key Concepts

| Concept | Description |
|--------|-------------|
| **Principal Components** | Directions of maximum variance |
| **Eigenvalues** | Amount of variance in each direction |
| **Eigenvectors** | Direction vectors |
| **Explained Variance** | Variance captured by each PC |

---

## Production Issues

### 1. Choosing Number of Components

**Problem**: Unknown optimal number

**Solutions**:
- Explained variance ratio
- Scree plot
- Cross-validation

### 2. Interpretability

**Problem**: Components hard to interpret

**Solutions**:
- Use fewer components
- Rotation (Varimax)
- Domain knowledge

### 3. Data Preprocessing

**Problem**: PCA sensitive to scaling

**Solutions**:
- Standardize features
- Handle missing values first

---

## Short Q&A

| Question | Answer |
|----------|--------|
| **What is the difference between PCA and LDA?** | PCA unsupervised (max variance), LDA supervised (max class separation) |
| **When should you use PCA?** | High dimensionality, noise reduction, visualization |
| **What are the limitations?** | Linear assumptions, information loss, interpretability |
| **How many components to keep?** | Keep components with 95% explained variance |
| **Is PCA supervised or unsupervised?** | Unsupervised - no target variable |
| **What is incremental PCA?** | For large datasets, processes in batches |

---

## Follow-up Questions

### How would you apply PCA for dimensionality reduction?

```
1. Preprocessing
   - Standardize features
   - Handle missing values

2. PCA Application
   - Fit PCA on training data
   - Transform both train and test
   - Choose components (95% variance)

3. Use Cases
   - Before classification
   - For visualization
   - For noise reduction
```
