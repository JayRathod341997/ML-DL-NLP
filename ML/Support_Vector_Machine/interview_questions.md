# Support Vector Machine - Interview Questions

## Basic Questions

### Q1: What is a Support Vector Machine?
**Answer:** SVM is a supervised learning algorithm that finds the optimal hyperplane that maximally separates different classes. It maximizes the margin (gap) between classes while minimizing classification errors.

### Q2: What are Support Vectors?
**Answer:** Support vectors are the data points that lie closest to the decision boundary (hyperplane). These are the critical points - if they are removed, the position of the hyperplane would change. They "support" the hyperplane.

### Q3: What is the margin in SVM?
**Answer:** The margin is the distance between the decision boundary (hyperplane) and the nearest data points (support vectors). SVM maximizes this margin - a larger margin generally leads to better generalization.

### Q4: What is a hyperplane?
**Answer:** A hyperplane is the decision boundary that separates classes:
- In 2D: A line
- In 3D: A plane
- In n-dimensions: An n-1 dimensional hyperplane

### Q5: What is the kernel trick?
**Answer:** The kernel trick allows SVM to find non-linear decision boundaries without explicitly transforming data. It computes the similarity (dot product) in a higher-dimensional space implicitly.

## Intermediate Questions

### Q6: What are the different SVM kernels?
**Answer:**
- **Linear:** For linearly separable data
- **Polynomial:** Creates polynomial decision boundaries
- **RBF (Radial Basis Function):** Creates circular/spherical boundaries
- **Sigmoid:** Similar to neural network activation

### Q7: What is the difference between C and gamma parameters?
**Answer:**
- **C (Regularization):** Controls trade-off between margin size and misclassification
  - Large C: Smaller margin, fewer misclassifications (overfit risk)
  - Small C: Larger margin, more misclassifications (underfit risk)
- **Gamma (RBF kernel):** Defines influence of each training example
  - Large gamma: Only close points affect decision boundary (complex)
  - Small gamma: Distant points also affect (simpler)

### Q8: Why is feature scaling important for SVM?
**Answer:** SVM uses distance calculations. Features with larger scales dominate, leading to biased results. Scaling ensures all features contribute equally.

### Q9: How does SVM handle multi-class classification?
**Answer:** Two approaches:
- **One-vs-Rest (OvR):** Train one classifier per class vs. all others
- **One-vs-One (OvO):** Train one classifier per pair of classes

### Q10: When should you use SVM?
**Answer:**
- When data has clear margin of separation
- For high-dimensional data
- When you need good generalization
- For text classification
- When number of features > number of samples

## Advanced Questions

### Q11: What is the difference between hard margin and soft margin SVM?
**Answer:**
- **Hard margin:** Strictly separates classes, only works with linearly separable data
- **Soft margin:** Allows some misclassifications, works with nearly separable data (uses C parameter)

### Q12: What are the advantages and disadvantages of SVM?
**Answer:**
**Advantages:**
- Effective in high dimensions
- Memory efficient (uses support vectors)
- Works well with clear margin
- Versatile with kernels

**Disadvantages:**
- Doesn't scale well to large datasets
- Sensitive to scaling
- Doesn't directly provide probabilities
- Doesn't work well when classes overlap

### Q13: What is the difference between SVM and Logistic Regression?
**Answer:**
| SVM | Logistic Regression |
|-----|-------------------|
| Maximizes margin | Maximizes likelihood |
| Uses support vectors | Uses all points |
| Decision boundary only | Provides probabilities |
| Kernel for non-linear | Limited to linear (without kernel) |
| Binary by nature | Naturally multi-class |

### Q14: How do you choose the right kernel?
**Answer:**
- Start with linear kernel
- Try RBF if data is not linearly separable
- Use cross-validation to compare kernels
- Consider data characteristics:
  - Linear: Text data, many features
  - RBF: Complex boundaries, moderate samples
  - Polynomial: When degree matters

### Q15: What happens when C approaches infinity in SVM?
**Answer:** The optimization problem becomes equivalent to hard margin SVM. The algorithm tries to classify all training points correctly, which can lead to overfitting and may fail if data is not linearly separable.

## Scenario-Based Questions

### Q16: Your SVM model is taking too long to train. What might be wrong?
**Answer:**
- Dataset is too large (SVM is O(n²) to O(n³))
- Using wrong kernel (RBF is slower)
- C is too large (more iterations needed)
- Consider using LinearSVC or SGDClassifier

### Q17: SVM gives different results on different runs. Why?
**Answer:** This can happen if:
- Using SVC (not LinearSVC) which uses random seeds
- Data has duplicate support vectors
- Need to set random_state parameter

### Q18: How do you handle imbalanced data in SVM?
**Answer:**
1. Use class_weight parameter
2. Use balanced weighting
3. Adjust decision threshold
4. Consider using SMOTE
5. Use appropriate evaluation metrics (F1, AUC)
