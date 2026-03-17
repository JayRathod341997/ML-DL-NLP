# Decision Tree - Interview Questions & System Design

## Fundamentals

### What is Decision Tree?

A supervised learning algorithm that creates a tree-like model of decisions based on feature values.

### How does Decision Tree work?

1. Start with all data at root
2. Find best feature to split
3. Create child nodes
4. Recursively split until stopping criteria

### Splitting Criteria

| Criterion | Description |
|-----------|-------------|
| **Gini Impurity** | Measures impurity (1 - Σp²) |
| **Entropy** | Information gain measure |
| **Variance Reduction** | For regression trees |

---

## System Design

### Tree Building Process

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Root      │────▶│   Feature  │────▶│   Split    │
│   Node      │     │   Selection│     │   Decision │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                    ┌──────────────────────────┘
                    ▼
            ┌─────────────┐
            │   Recursive │
            │   Branching │
            └─────────────┘
```

---

## Production Issues

### 1. Overfitting

**Problem**: Tree too complex, poor generalization

**Solutions**:
- Limit tree depth (max_depth)
- Minimum samples to split
- Minimum samples in leaf
- Pruning (cost-complexity)

### 2. Unstable Trees

**Problem**: Small data changes cause different trees

**Solutions**:
- Ensemble methods (Random Forest)
- Use bootstrap sampling
- Set proper constraints

### 3. Feature Selection Bias

**Problem**: Favors features with many values

**Solutions**:
- Use information gain ratio
- Feature engineering
- Preprocessing

---

## Short Q&A

| Question | Answer |
|----------|--------|
| **What is Gini impurity?** | Measure of node purity: 1 - Σp² |
| **What is entropy?** | Information content: -Σp × log(p) |
| **What is information gain?** | Entropy before split - weighted entropy after |
| **When to use Decision Tree?** | When interpretability needed, categorical features |
| **What is pruning?** | Removing branches to reduce complexity |
| **Difference between ID3, C4.5, CART?** | ID3 uses entropy, C4.5 uses gain ratio, CART uses Gini |
| **What is the time complexity?** | O(n × d × log(n)) for building, O(d) for prediction |

---

## Follow-up Questions

### How would you prevent overfitting in Decision Trees?

```
1. Pre-pruning
   - Limit max_depth
   - Minimum samples to split
   - Minimum samples in leaf

2. Post-pruning
   - Cost-complexity pruning
   - Reduced error pruning

3. Ensemble Methods
   - Random Forest (bagging)
   - Gradient Boosting (boosting)
```

### How do you handle categorical variables in Decision Trees?

```
1. Ordinal Encoding
   - For ordered categories

2. One-Hot Encoding
   - For nominal categories

3. Native Support
   - Some implementations handle natively
```
