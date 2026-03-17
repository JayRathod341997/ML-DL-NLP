# Production Issues and Solutions

## Common Production Issues for K-Means Customer Segmentation

### 1. Cluster Drift
**Problem**: Customer segments change over time as behavior shifts.
**Solution**: Retrain model periodically and monitor cluster stability.

### 2. Feature Engineering
**Problem**: Inconsistent feature computation between training and inference.
**Solution**: Use feature store and consistent preprocessing pipelines.

### 3. Scaling Issues
**Problem**: Features must be scaled consistently.
**Solution**: Save scaler with model and apply same scaling to new data.

### 4. Choosing Optimal K
**Problem**: Wrong number of clusters leads to poor segmentation.
**Solution**: Use elbow method, silhouette score, and business requirements.

### 5. Empty Clusters
**Problem**: Some clusters may become empty with new data.
**Solution**: Monitor cluster sizes and retrain when needed.
