# K-Means Clustering - Interview Questions

## Basic Questions

### Q1: What is K-Means clustering?
**Answer:** K-Means is an unsupervised learning algorithm that partitions data into K clusters. It aims to minimize the within-cluster variance (sum of squared distances from points to their cluster centroids).

### Q2: How does the K-Means algorithm work?
**Answer:**
1. Initialize K centroids (randomly or using K-Means++)
2. Assign each point to the nearest centroid
3. Recalculate centroids as the mean of assigned points
4. Repeat steps 2-3 until convergence or max iterations

### Q3: What is the objective function of K-Means?
**Answer:** Minimize the within-cluster sum of squares (WCSS):
```
J = Σᵢ Σₓ ||xⱼ⁽ⁱ⁾ - μᵢ||²
```
where μᵢ is the centroid of cluster i.

### Q4: What is the difference between hard and soft clustering?
**Answer:**
- **Hard clustering:** Each point belongs to exactly one cluster
- **Soft clustering:** Each point has a probability of belonging to each cluster

K-Means is hard clustering.

### Q5: What are the limitations of K-Means?
**Answer:**
- Must specify K in advance
- Sensitive to initialization
- Assumes spherical clusters of similar size
- Sensitive to outliers
- Can get stuck in local optima

## Intermediate Questions

### Q6: What is the Elbow method?
**Answer:** The Elbow method helps find the optimal number of clusters by plotting WCSS vs. number of clusters. The "elbow" point where the rate of decrease sharply changes indicates the optimal K.

### Q7: What is K-Means++ initialization?
**Answer:** K-Means++ improves centroid initialization:
1. Choose first centroid randomly
2. For each subsequent centroid, choose point with probability proportional to distance squared
3. Run standard K-Means
This leads to better convergence and quality.

### Q8: How do you choose the value of K?
**Answer:**
1. Elbow method
2. Silhouette analysis
3. Gap statistic
4. Domain knowledge
5. Cross-validation approaches

### Q9: What is inertia in K-Means?
**Answer:** Inertia is the sum of squared distances from each point to its assigned centroid. Lower inertia means tighter clusters.

### Q10: How does K-Means handle categorical data?
**Answer:** Standard K-Means doesn't handle categorical data directly. Options:
- Use K-Modes algorithm for categorical data
- One-hot encode categorical features
- Use other algorithms (DBSCAN, hierarchical clustering)

## Advanced Questions

### Q11: What is the difference between K-Means and KNN?
**Answer:**
| K-Means | KNN |
|---------|-----|
| Unsupervised | Supervised |
| Clustering | Classification |
| No labels needed | Requires labels |
| Finds clusters | Classifies points |
| K = clusters | K = neighbors |

### Q12: How do you evaluate K-Means clustering?
**Answer:**
- Silhouette score: Measures how similar points are to own cluster vs other clusters
- Davies-Bouldin index: Ratio of within-cluster to between-cluster distances
- Calinski-Harabasz index: Ratio of between-cluster to within-cluster variance
- Manual inspection (domain knowledge)

### Q13: What are the advantages of K-Means?
**Answer:**
- Simple and easy to understand
- Scalable to large datasets
- Fast convergence
- Parameters are interpretable (K, max_iter)
- Works well when clusters are spherical

### Q14: What is Mini-Batch K-Means?
**Answer:** Mini-Batch K-Means uses small random batches of data instead of the full dataset for each iteration. It's faster for large datasets with similar results to standard K-Means.

### Q15: How do you handle outliers in K-Means?
**Answer:**
1. Remove outliers before clustering
2. Use K-Medoids (PAM) instead of K-Means
3. Use robust initialization
4. Run multiple times with different initializations
5. Consider using DBSCAN or hierarchical clustering

## Scenario-Based Questions

### Q16: Your K-Means clusters are very uneven. What might be the cause?
**Answer:**
- Data may have natural imbalance
- K may not match data structure
- Initialization issues
- Consider using different K
- Try K-Means++ initialization

### Q17: K-Means gives different results on different runs. Why?
**Answer:** Random initialization! Use:
- K-Means++ initialization
- Set random_state for reproducibility
- Run multiple times and pick best (lowest inertia)
