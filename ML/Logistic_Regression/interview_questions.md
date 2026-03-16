# Logistic Regression - Interview Questions

## Basic Questions

### Q1: What is Logistic Regression?
**Answer:** Logistic Regression is a supervised learning algorithm used for classification problems. Despite its name, it's a classification algorithm, not regression. It predicts the probability of a binary outcome using the sigmoid function.

### Q2: Why is it called "Logistic Regression" if it's used for classification?
**Answer:** It's called "regression" because it uses the same underlying mathematical principles as linear regression (finding coefficients), but applies a logistic function to output probabilities for classification.

### Q3: What is the sigmoid function? Write its formula.
**Answer:** The sigmoid function maps any real number to a value between 0 and 1:
```
σ(z) = 1 / (1 + e^(-z))
```
where z = β₀ + β₁x₁ + β₂x₂ + ...

### Q4: What is the difference between Logistic Regression and Linear Regression?
**Answer:**
| Feature | Linear Regression | Logistic Regression |
|---------|------------------|-------------------|
| Purpose | Predict continuous values | Predict classes |
| Output | Any number | Probability (0-1) |
| Assumption | Linear relationship | Binary outcome |
| Loss Function | MSE | Log Loss |

### Q5: What is the log-odds (logit)?
**Answer:** Log-odds is the natural log of the odds of the probability:
```
logit(p) = log(p / (1 - p))
```
This is what Logistic Regression actually models - the log-odds as a linear function of features.

## Intermediate Questions

### Q6: What is the loss function used in Logistic Regression?
**Answer:** Log Loss (Cross-Entropy Loss):
```
L = -[y log(ŷ) + (1-y) log(1-ŷ)]
```
This penalizes confident wrong predictions more than uncertain ones.

### Q7: What is Maximum Likelihood Estimation (MLE)?
**Answer:** MLE finds the coefficients that maximize the likelihood of observing the given data. Instead of minimizing error (like MSE), it maximizes the probability of the data given the model.

### Q8: How do you handle multi-class classification in Logistic Regression?
**Answer:** Two approaches:
1. **One-vs-Rest (OvR):** Train one classifier per class vs all others
2. **Softmax (Multinomial):** Directly model the probability of each class

### Q9: What are the assumptions of Logistic Regression?
**Answer:**
1. Binary target variable (for binary classification)
2. Observations are independent
3. No multicollinearity among features
4. Large sample size
5. Linear relationship between features and log-odds

### Q10: How do you evaluate Logistic Regression models?
**Answer:**
- Accuracy
- Precision
- Recall/F1-Score
- AUC-ROC
- Confusion Matrix
- Log Loss

## Advanced Questions

### Q11: What is the difference between Precision and Recall?
**Answer:**
- **Precision:** Of all predicted positives, how many are actually positive?
  - TP / (TP + FP)
- **Recall:** Of all actual positives, how many did we predict?
  - TP / (TP + FN)

### Q12: What is the AUC-ROC curve?
**Answer:** AUC-ROC (Area Under the Receiver Operating Characteristic Curve) measures the model's ability to distinguish between classes. AUC = 1 means perfect classification, AUC = 0.5 means random guessing.

### Q13: How do you interpret Logistic Regression coefficients?
**Answer:** 
- Positive coefficient: Increases probability of positive class
- Negative coefficient: Decreases probability of positive class
- The coefficient represents the change in log-odds for a one-unit change in the feature

### Q14: What is regularization in Logistic Regression? Why is it used?
**Answer:** Regularization (L1/L2) adds a penalty term to the loss function to prevent overfitting. It shrinks coefficients towards zero.

### Q15: What is the difference between L1 and L2 regularization in Logistic Regression?
**Answer:**
- L2 (Ridge): Shrinks coefficients but keeps all features
- L1 (Lasso): Can shrink coefficients to exactly zero (feature selection)

## Scenario-Based Questions

### Q16: Your model has high accuracy but low recall. What does this mean?
**Answer:** The model is not catching enough positive cases. It's being too conservative in predicting positive. Solutions:
- Lower the threshold (from 0.5 to 0.3)
- Use class weights
- Resample data

### Q17: How would you handle imbalanced classes in Logistic Regression?
**Answer:**
1. Use class weights
2. Oversample minority class / Undersample majority
3. Use SMOTE
4. Change evaluation metric (use F1 instead of accuracy)
5. Adjust threshold

### Q18: When would you choose Logistic Regression over other classifiers?
**Answer:** Choose Logistic Regression when:
- You need probabilistic outputs
- You want interpretable models
- Data is linearly separable
- You need fast training
- Baseline model is needed
