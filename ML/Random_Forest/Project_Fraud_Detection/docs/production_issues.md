# Production Issues and Solutions

## Common Production Issues for Random Forest

### 1. Overfitting
**Problem:** High training accuracy but poor test performance.
**Solution:** Limit tree depth, increase min_samples_split, use cross-validation.

### 2. Class Imbalance
**Problem:** Poor performance on minority class (fraud detection).
**Solution:** Use class_weight='balanced', SMOTE, or adjust threshold.

### 3. Memory Issues
**Problem:** Large number of trees consumes memory.
**Solution:** Reduce n_estimators, use max_features, or consider incremental learning.

### 4. Slow Predictions
**Problem:** Inference is slow with many trees.
**Solution:** Reduce number of trees in production, use model compression.

### 5. Feature Importance
**Problem:** Misleading feature importance with correlated features.
**Solution:** Use permutation importance, remove highly correlated features.
