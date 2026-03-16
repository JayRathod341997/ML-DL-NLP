# Principal Component Analysis - Interview Questions

## Basic Questions

### Q1: What is PCA?
**Answer:** PCA (Principal Component Analysis) is a dimensionality reduction technique that transforms data to a new coordinate system. It identifies directions (principal components) where data varies the most and projects the data onto these components.

### Q2: What are principal components?
**Answer:** Principal components are orthogonal axes (directions) in the feature space that represent the directions of maximum variance. The first component captures the most variance, the second captures the second most, and so on.

### Q3: Why do we need to standardize data before PCA?
**Answer:** PCA is sensitive to the scale of features. Features with larger scales will dominate the variance calculation. Standardization (mean=0, std=1) ensures all features contribute equally.

### Q4: What is the difference between PCA and dimensionality reduction?
**Answer:** PCA is one method of dimensionality reduction, but not all dimensionality reduction uses PCA. Other methods include:
- t-SNE
- UMAP
- Autoencoders
- Feature selection

### Q5: How does PCA work?
**Answer:**
1. Standardize the data
2. Compute the covariance matrix
3. Calculate eigenvalues and eigenvectors
4. Sort by eigenvalues (descending)
5. Choose top k components
6. Project data onto these components

## Intermediate Questions

### Q6: What is explained variance?
**Answer:** Explained variance is the amount of total variance in the data that each principal component captures. It's usually expressed as a percentage. Sum of all explained variances = 100%.

### Q7: How do you choose the number of components?
**Answer:**
- **Scree plot:** Look for the "elbow" where adding more components doesn't help much
- **Explained variance:** Keep components that explain >70-90% variance
- **Cross-validation:** Test performance on downstream task
- ** Kaiser's rule:** Keep components with eigenvalue > 1

### Q8: What is the difference between PCA and SVD?
**Answer:** PCA and SVD are mathematically related. SVD is more numerically stable and can be used directly. PCA is often implemented using SVD under the hood.

### Q9: What are the applications of PCA?
**Answer:**
- Dimensionality reduction
- Data visualization (2D/3D plots)
- Noise reduction
- Feature extraction
- Preprocessing for other algorithms
- Face recognition (Eigenfaces)

### Q10: What are the limitations of PCA?
**Answer:**
- Assumes linear relationships
- Sensitive to outliers
- Components may not be interpretable
- Loses some information (by design)
- Can't handle non-linear patterns

## Advanced Questions

### Q11: What is the relationship between PCA and eigenvalues/eigenvectors?
**Answer:** PCA computes eigenvalues and eigenvectors of the covariance matrix. Each eigenvector is a principal component, and its eigenvalue represents the variance explained by that component.

### Q12: What is the difference between PCA and Factor Analysis?
**Answer:**
- **PCA:** Maximizes variance, components are linear combinations of all features
- **Factor Analysis:** Maximizes shared variance, assumes latent factors exist

### Q13: When should you use PCA?
**Answer:**
- Too many features (curse of dimensionality)
- Visualizing high-dimensional data
- Speeding up other algorithms
- Removing multicollinearity
- Noise reduction
- As preprocessing for ML algorithms

### Q14: What is incremental PCA?
**Answer:** Incremental PCA processes data in batches instead of loading all data into memory. It's useful for large datasets that can't fit in memory.

### Q15: Can PCA be used for supervised learning?
**Answer:** PCA itself is unsupervised (doesn't use labels). However, it's commonly used as a preprocessing step for supervised learning:
1. Apply PCA to reduce dimensions
2. Use reduced features for classification/regression

## Scenario-Based Questions

### Q16: Your PCA components explain only 50% of variance with 10 components. What does this mean?
**Answer:** This suggests:
- Data has high intrinsic dimensionality
- Features may be correlated in complex ways
- You might need more components
- Or the data might not be suitable for linear PCA

### Q17: PCA gives different results on different runs. Why?
**Answer:**
- If using randomized SVD
- Data not properly standardized
- Floating point precision issues
- Set random_state for reproducibility

### Q18: When would you NOT use PCA?
**Answer:**
- When interpretability is crucial
- When data has non-linear relationships
- When you need to keep original features
- For categorical data (use MCA or other methods)
- When computation time is a constraint for simple problems
