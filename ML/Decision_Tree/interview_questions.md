# Decision Tree - Interview Questions

## Basic Questions

### Q1: What is a Decision Tree?
**Answer:** A Decision Tree is a supervised learning algorithm that makes decisions by splitting data based on feature values. It uses a tree-like structure where each internal node represents a test on a feature, each branch represents the outcome, and each leaf node represents a class label or prediction.

### Q2: What are the components of a Decision Tree?
**Answer:**
- **Root Node:** The topmost node, representing the entire dataset and the first split
- **Internal Nodes:** Nodes that represent tests on features
- **Branches:** Outcomes of the splits (yes/no paths)
- **Leaf Nodes:** Terminal nodes that provide the final prediction

### Q3: What is Gini Impurity?
**Answer:** Gini Impurity measures the probability of incorrectly classifying a randomly chosen element if it was labeled according to the class distribution. It's calculated as:
```
Gini = 1 - Σ(p_i)²
```
where p_i is the probability of class i. Lower Gini = purer node.

### Q4: What is Entropy in Decision Trees?
**Answer:** Entropy measures the disorder or uncertainty in the data. It's calculated as:
```
Entropy = -Σ(p_i * log₂(p_i))
```
Information Gain = Entropy(before split) - Weighted Entropy(after split)

### Q5: What is Information Gain?
**Answer:** Information Gain measures the reduction in entropy achieved by splitting the data on a feature. The feature with the highest Information Gain is chosen for splitting.

## Intermediate Questions

### Q6: What is the difference between Gini and Entropy?
**Answer:**
- Gini ranges from 0 to 0.5 (for binary classification)
- Entropy ranges from 0 to 1
- Both are used to measure impurity and make splitting decisions
- Gini is computationally simpler (no logarithms)
- Results are usually similar

### Q7: What is pruning? Why is it needed?
**Answer:** Pruning is the process of removing nodes from the tree to reduce complexity and prevent overfitting. Types:
- **Pre-pruning (Early stopping):** Stop growing tree early
- **Post-pruning:** Grow full tree, then remove nodes

### Q8: What are the hyperparameters in Decision Trees?
**Answer:**
- max_depth: Maximum depth of tree
- min_samples_split: Minimum samples to split a node
- min_samples_leaf: Minimum samples in a leaf
- max_features: Number of features to consider
- criterion: gini or entropy

### Q9: How do Decision Trees handle categorical variables?
**Answer:** Decision Trees can handle categorical variables directly. For a categorical feature with k categories, the split creates k branches. Some implementations encode categories numerically.

### Q10: Can Decision Trees handle missing values?
**Answer:** Most Decision Tree implementations have built-in handling for missing values:
- Assign to most common branch
- Use surrogate splits
- Some algorithms allow missing as a separate category

## Advanced Questions

### Q11: What is the difference between ID3, C4.5, and CART algorithms?
**Answer:**
- **ID3:** Uses Information Gain, only categorical features
- **C4.5:** Uses Information Gain Ratio, handles continuous features
- **CART:** Uses Gini, produces binary trees, can do regression

### Q12: What are the advantages and disadvantages of Decision Trees?
**Answer:**
**Advantages:**
- Easy to interpret
- Can handle both numerical and categorical data
- Requires little data preprocessing
- Handles missing values

**Disadvantages:**
- Prone to overfitting
- Can create biased trees with imbalanced data
- Small changes in data can lead to different trees
- Greedy algorithm may not find optimal tree

### Q13: How do you prevent overfitting in Decision Trees?
**Answer:**
1. Limit tree depth (max_depth)
2. Require minimum samples to split (min_samples_split)
3. Require minimum samples in leaf (min_samples_leaf)
4. Limit number of features (max_features)
5. Pruning the tree

### Q14: Can Decision Trees be used for regression?
**Answer:** Yes! Decision Trees can be used for regression (Decision Tree Regressor). Instead of Gini/Entropy, they use MSE or MAE as the splitting criterion. The prediction is the mean of values in the leaf.

### Q15: What is the difference between Classification and Regression Decision Trees?
**Answer:**
| Aspect | Classification Tree | Regression Tree |
|--------|-------------------|-----------------|
| Target | Categorical | Continuous |
| Split Criterion | Gini/Entropy | MSE/MAE |
| Prediction | Class label | Numerical value |
| Leaf Value | Majority class | Mean of values |

## Scenario-Based Questions

### Q16: Your Decision Tree is overfitting. What will you do?
**Answer:**
1. Reduce max_depth
2. Increase min_samples_split
3. Increase min_samples_leaf
4. Use pruning
5. Reduce number of features
6. Collect more data

### Q17: Which is more important: Gini or Information Gain?
**Answer:** Neither is universally better. Both produce similar results in practice. Gini is computationally faster (no logarithms), while Information Gain can be biased towards features with many categories.

### Q18: How do you handle imbalanced data in Decision Trees?
**Answer:**
1. Use class_weight parameter
2. Use sampling techniques (SMOTE)
3. Adjust min_samples_leaf by class
4. Use different evaluation metrics (F1, AUC)
5. Prune differently for minority class
