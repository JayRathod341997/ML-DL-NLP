# Support Vector Machine - Interview Questions & System Design

## Fundamentals

### What is SVM?

Support Vector Machine is a supervised learning algorithm that finds the optimal hyperplane to separate classes in feature space.

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Hyperplane** | Decision boundary that separates classes |
| **Support Vectors** | Data points closest to hyperplane |
| **Margin** | Distance between hyperplane and support vectors |
| **Kernel** | Function to transform data to higher dimension |

### Kernel Types

1. **Linear**: For linearly separable data
2. **Polynomial**: For curved boundaries
3. **RBF (Gaussian)**: For complex boundaries
4. **Sigmoid**: Similar to neural network

---

## System Design

### SVM Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Input     │────▶│   Kernel    │────▶│   Optimal   │
│   Features  │     │   Transform │     │   Hyperplane│
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │   Class     │
                                        │   Prediction│
                                        └─────────────┘
```

---

## Production Issues

### 1. Training Time

**Problem**: O(n²) to O(n³) complexity

**Solutions**:
- Use LinearSVC for large datasets
- Subsample training data
- Use SGDClassifier

### 2. Kernel Selection

**Problem**: Wrong kernel leads to poor performance

**Solutions**:
- Cross-validation
- Start with RBF kernel
- Consider data characteristics

### 3. Parameter Tuning

**Problem**: C and gamma sensitive

**Solutions**:
- Grid search with cross-validation
- Use randomized search
- Start with default and adjust

---

## Short Q&A

| Question | Answer |
|----------|--------|
| **What is the role of C parameter?** | Regularization - controls trade-off between margin and misclassification |
| **What is gamma in RBF kernel?** | Controls influence of single training example |
| **What are support vectors?** | Data points on the margin boundaries |
| **Why is SVM effective in high dimensions?** | Uses kernel trick, doesn't suffer from curse of dimensionality |
| **Difference between hard and soft margin?** | Hard - strict separation; Soft - allows some misclassification |
| **When to use SVM over other algorithms?** | High-dimensional data, clear margin separation, text classification |
| **What is the kernel trick?** | Computing in higher dimension without explicit transformation |

---

## Follow-up Questions

### How would you optimize SVM for production?

```
1. Algorithm Selection
   - LinearSVC for large n
   - SGDClassifier for very large datasets
   - Use libsvm for small datasets

2. Feature Engineering
   - Feature scaling essential
   - Dimensionality reduction if needed
   - Handle categorical variables

3. Parameter Tuning
   - Grid search for C and gamma
   - Use logarithmic scale for gamma
   - Balance bias-variance
```
