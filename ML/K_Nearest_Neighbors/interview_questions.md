# K-Nearest Neighbors - Interview Questions

## Basic Questions

### Q1: What is K-Nearest Neighbors (KNN)?
**Answer:** KNN is a simple, instance-based learning algorithm that classifies new data points based on the majority class of their K nearest neighbors in the training data. It doesn't have an explicit training phase - it simply stores the training data.

### Q2: Why is KNN called a "lazy" algorithm?
**Answer:** KNN is called "lazy" because it doesn't learn a discriminative function from training data. Instead, it memorizes the entire training dataset and performs all computation at prediction time.

### Q3: How does KNN make predictions?
**Answer:**
1. Calculate distance between the new point and all training points
2. Find the K nearest neighbors
3. For classification: Take majority vote of neighbor classes
4. For regression: Average the values of K neighbors

### Q4: What are the different distance metrics used in KNN?
**Answer:**
- Euclidean distance: √(Σ(xᵢ-yᵢ)²)
- Manhattan distance: Σ|xᵢ-yᵢ|
- Minkowski distance: (Σ|xᵢ-yᵢ|^p)^(1/p)
- Hamming distance: For categorical features

### Q5: How do you choose the value of K?
**Answer:**
- Small K: Sensitive to noise, can overfit
- Large K: Smoother decision boundary, can underfit
- Use cross-validation to find optimal K
- Odd K for binary classification to avoid ties

## Intermediate Questions

### Q6: What is the curse of dimensionality in KNN?
**Answer:** As the number of dimensions (features) increases, the distance between points becomes less meaningful. In high dimensions, all points become equally distant. This makes KNN less effective.

### Q7: Why is feature scaling important in KNN?
**Answer:** KNN uses distance calculations. Features with larger scales dominate the distance, making the algorithm biased. Feature scaling (normalization or standardization) ensures all features contribute equally.

### Q8: What are the advantages of KNN?
**Answer:**
- Simple to understand and implement
- No training phase (fast to set up)
- Naturally handles multi-class classification
- No assumptions about data distribution
- Can be used for both classification and regression

### Q9: What are the disadvantages of KNN?
**Answer:**
- Slow prediction for large datasets
- Sensitive to irrelevant features
- Sensitive to noise and outliers
- Requires feature scaling
- Memory-intensive (stores all training data)
- Doesn't work well with high-dimensional data

### Q10: How does K handle imbalanced data?
**Answer:**
- Use weighted voting (closer neighbors have more weight)
- Use stratified sampling for neighbors
- Use appropriate K values
- Consider using different metrics

## Advanced Questions

### Q11: What is the difference between weighted and unweighted KNN?
**Answer:**
- **Unweighted:** Each neighbor has equal vote
- **Weighted:** Closer neighbors have more influence (typically 1/distance)

### Q12: How do you handle missing values in KNN?
**Answer:**
1. Impute missing values before using KNN
2. Use distance metrics that handle missing values
3. Ignore missing values in distance calculation
4. Use KNNImputer for imputation

### Q13: When should you use KNN?
**Answer:**
- Small to medium datasets
- When interpretability is important
- As a baseline model
- When decision boundary is complex
- When data has meaningful distance metrics

### Q14: How does KNN work for regression?
**Answer:** For regression, KNN predicts a continuous value by averaging the target values of the K nearest neighbors. Weighted averaging can also be used.

### Q15: What is the difference between KNN and K-Means?
**Answer:**
| KNN | K-Means |
|-----|---------|
| Supervised learning | Unsupervised learning |
| Uses labeled data | Uses unlabeled data |
| Classification/Regression | Clustering |
| K = number of neighbors | K = number of clusters |
| Lazy learner | Eager learner |
