# Random Forest - Interview Questions

## Basic Questions

### Q1: What is Random Forest?
**Answer:** Random Forest is an ensemble learning algorithm that builds multiple Decision Trees during training and outputs the class that is the mode of the classes (for classification) or mean prediction (for regression) of individual trees. It provides better accuracy and reduces overfitting compared to single Decision Trees.

### Q2: What is the difference between Bagging and Random Forest?
**Answer:** Bagging (Bootstrap Aggregating) builds multiple models on bootstrap samples and averages their predictions. Random Forest is a specific type of bagging that:
- Uses Decision Trees as base learners
- Introduces additional randomness by selecting random subsets of features at each split

### Q3: What is bootstrapping?
**Answer:** Bootstrapping is the process of creating multiple subsets of the original dataset by randomly sampling with replacement. Each tree in Random Forest is trained on a different bootstrap sample.

### Q4: Why is Random Forest called "Random"?
**Answer:** Two sources of randomness:
1. Each tree is trained on a different bootstrap sample
2. At each node split, only a random subset of features is considered

### Q5: How does Random Forest handle overfitting?
**Answer:** By combining multiple trees trained on different data samples:
- Individual trees may overfit
- Averaging predictions reduces variance
- More trees = less overfitting

## Intermediate Questions

### Q6: What are the key hyperparameters in Random Forest?
**Answer:**
- n_estimators: Number of trees
- max_depth: Maximum depth of trees
- min_samples_split: Minimum samples to split
- min_samples_leaf: Minimum samples in leaf
- max_features: Features to consider at each split
- bootstrap: Whether to use bootstrap samples

### Q7: What is Out-of-Bag (OOB) error?
**Answer:** OOB error is the prediction error calculated using samples that were not included in a particular tree's bootstrap sample. It's a built-in cross-validation method - no separate validation set needed!

### Q8: What is Feature Importance in Random Forest?
**Answer:** Feature Importance measures how much each feature contributes to the model's predictions. It's calculated by:
1. Measuring how much each feature decreases impurity (Gini Importance)
2. Or measuring accuracy decrease when feature is removed (Permutation Importance)

### Q9: How do you tune Random Forest?
**Answer:**
1. Start with default parameters
2. Tune n_estimators (more trees = better but slower)
3. Tune max_depth and min_samples
4. Use OOB error or cross-validation
5. Consider max_features for optimal splits

### Q10: What is the difference between Random Forest and Gradient Boosting?
**Answer:**
| Random Forest | Gradient Boosting |
|--------------|-------------------|
| Parallel tree building | Sequential tree building |
| Trees are independent | Each tree learns from previous errors |
| Reduces variance | Reduces bias |
| Easier to tune | More hyperparameters |
| Less prone to overfitting | Can overfit if not tuned |

## Advanced Questions

### Q11: When should you NOT use Random Forest?
**Answer:**
- When interpretability is crucial (use single Decision Tree)
- When prediction speed is critical (more trees = slower)
- For very high-dimensional sparse data (like text)
- When you need probability calibration

### Q12: How does Random Forest handle missing values?
**Answer:**
1. Can handle missing values natively
2. Uses surrogate splits (alternative features)
3. Missing values sent to majority vote in leaf

### Q13: What is the relationship between number of trees and performance?
**Answer:** Performance generally improves with more trees, but:
- Diminishing returns after ~100-300 trees
- More trees = more memory and time
- With enough trees, adding more doesn't hurt

### Q14: How do you handle imbalanced data in Random Forest?
**Answer:**
1. Use class_weight='balanced'
2. Set sample_weight
3. Use stratified sampling
4. Focus on F1/AUC metrics
5. Consider SMOTE

### Q15: What are the advantages and disadvantages of Random Forest?
**Answer:**
**Advantages:**
- High accuracy
- Handles missing values
- Provides feature importance
- Works well with default parameters
- Less prone to overfitting

**Disadvantages:**
- Slower than single tree
- Less interpretable
- Can be large in memory
- May not perform well on very imbalanced data
