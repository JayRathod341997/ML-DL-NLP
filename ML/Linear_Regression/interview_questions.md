# Linear Regression - Interview Questions

## Basic Questions

### Q1: What is Linear Regression?
**Answer:** Linear Regression is a supervised learning algorithm used for predicting a continuous (numerical) target variable based on one or more input features. It finds the best-fit straight line (or plane) that minimizes the sum of squared errors between predicted and actual values.

### Q2: What is the difference between Simple and Multiple Linear Regression?
**Answer:** Simple Linear Regression uses only one input feature to predict the target, while Multiple Linear Regression uses two or more features.

### Q3: What are the assumptions of Linear Regression?
**Answer:**
1. Linearity - Relationship between X and Y is linear
2. Independence - Observations are independent
3. Homoscedasticity - Constant variance of errors
4. Normality - Errors are normally distributed
5. No multicollinearity (for multiple regression)

### Q4: How do you interpret the coefficients in Linear Regression?
**Answer:** The coefficient (slope) represents the change in Y for each one-unit change in X. For example, if coefficient = 5, then Y increases by 5 units for every 1 unit increase in X.

### Q5: What is the intercept?
**Answer:** The intercept (b) is the value of Y when X equals zero. It's where the regression line crosses the Y-axis.

## Intermediate Questions

### Q6: What is R-squared? How do you interpret it?
**Answer:** R-squared (coefficient of determination) measures the proportion of variance in the dependent variable explained by the independent variables. It ranges from 0 to 1:
- 0 = Model explains none of the variance
- 1 = Model explains all of the variance
- Generally, higher R² is better, but not always!

### Q7: What is the difference between R-squared and Adjusted R-squared?
**Answer:** R-squared always increases when you add more features, even if they're not useful. Adjusted R-squared penalizes unnecessary features and only increases if the new feature improves the model significantly.

### Q8: What is Multicollinearity? How do you detect it?
**Answer:** Multicollinearity occurs when two or more independent features are highly correlated with each other. It can be detected using:
- Variance Inflation Factor (VIF)
- Correlation matrix
- Tolerance values

### Q9: How do you handle Multicollinearity?
**Answer:**
- Remove one of the correlated features
- Use regularization (Ridge, Lasso)
- Combine correlated features (PCA)
- Use centered data

### Q10: What is the difference between RMSE, MSE, and MAE?
**Answer:**
- MSE (Mean Squared Error): Average of squared differences
- RMSE (Root MSE): Square root of MSE (same unit as Y)
- MAE (Mean Absolute Error): Average of absolute differences

## Advanced Questions

### Q11: What is the difference between Ridge and Lasso regression?
**Answer:** Both are regularization techniques:
- Ridge (L2): Shrinks coefficients towards zero but rarely makes them exactly zero
- Lasso (L1): Can shrink coefficients exactly to zero (feature selection)

### Q12: How do you choose the regularization parameter (lambda)?
**Answer:** Use cross-validation to test different values and choose the one that gives the best performance on validation data.

### Q13: What is overfitting in Linear Regression? How do you prevent it?
**Answer:** Overfitting occurs when the model is too complex and learns noise in training data. Prevention:
- Use regularization (Ridge/Lasso)
- Reduce number of features
- Use cross-validation
- Collect more data

### Q14: When should you NOT use Linear Regression?
**Answer:**
- When relationship is not linear
- For classification problems
- When data has outliers
- When features are not independent

### Q15: How do you validate a Linear Regression model?
**Answer:**
- Split data into train/test sets
- Use k-fold cross-validation
- Check residual plots
- Compare metrics (R², RMSE, MAE)
- Check for assumption violations

## Scenario-Based Questions

### Q16: Your R-squared is 0.95 but model performs poorly on test data. What could be wrong?
**Answer:** This indicates overfitting! The model memorized training data. Solutions:
- Use regularization
- Reduce model complexity
- Get more training data
- Use cross-validation

### Q17: Two features have high correlation. What will happen?
**Answer:** Multicollinearity! This causes:
- Unstable coefficient estimates
- Inflated standard errors
- Incorrect interpretation of feature importance

### Q18: How would you improve a Linear Regression model?
**Answer:**
1. Check and transform features (log, sqrt)
2. Handle outliers
3. Add polynomial features for non-linear relationships
4. Use regularization
5. Feature selection
6. Address multicollinearity
