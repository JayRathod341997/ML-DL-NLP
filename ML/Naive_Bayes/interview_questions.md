# Naive Bayes - Interview Questions

## Basic Questions

### Q1: What is Naive Bayes?
**Answer:** Naive Bayes is a classification algorithm based on Bayes' theorem with the "naive" assumption of conditional independence between features. It calculates the probability of each class given the features and selects the class with the highest probability.

### Q2: What is Bayes' theorem?
**Answer:** Bayes' theorem calculates conditional probability:
```
P(Class|Features) = P(Features|Class) × P(Class) / P(Features)
```
Where:
- P(Class|Features): Posterior probability
- P(Features|Class): Likelihood
- P(Class): Prior probability

### Q3: Why is it called "Naive"?
**Answer:** It's called "naive" because it assumes that all features are conditionally independent of each other given the class. This assumption rarely holds in real data but simplifies calculations significantly.

### Q4: What are the types of Naive Bayes?
**Answer:**
- **Gaussian:** For continuous features (assumes normal distribution)
- **Multinomial:** For text/count data (word frequencies)
- **Bernoulli:** For binary features
- **Categorical:** For categorical features with multinomial distribution

### Q5: Why is Naive Bayes fast?
**Answer:**
- No iterative optimization
- Simple probability calculations
- Feature independence assumption simplifies computation
- Can train on small datasets quickly

## Intermediate Questions

### Q6: What are the advantages of Naive Bayes?
**Answer:**
- Very fast training and prediction
- Simple to understand
- Works well with high-dimensional data
- Handles missing values well
- Good baseline model

### Q7: What are the disadvantages of Naive Bayes?
**Answer:**
- Strong independence assumption (rarely true)
- Cannot learn relationships between features
- Zero probability problem (smoothing needed)
- May not perform well when features are correlated

### Q8: How do you handle the zero probability problem?
**Answer:** Use Laplace (additive) smoothing:
```
P(xᵢ|c) = (count(xᵢ,c) + α) / (count(c) + α×|V|)
```
where α is smoothing parameter, |V| is vocabulary size.

### Q9: When should you use Naive Bayes?
**Answer:**
- Text classification (spam detection, sentiment analysis)
- High-dimensional data
- As a baseline model
- When speed is important
- When independence assumption roughly holds

### Q10: What is the difference between Naive Bayes and Logistic Regression?
**Answer:**
| Naive Bayes | Logistic Regression |
|-------------|-------------------|
| Generative model | Discriminative model |
| Estimates P(Features|Class) | Estimates P(Class|Features) |
| Assumes feature independence | No independence assumption |
| Faster | More accurate generally |
| May underfit | Better fit to data |

## Advanced Questions

### Q11: Does Naive Bayes handle missing values?
**Answer:** Yes, it handles missing values naturally:
- For training: Skip the missing feature in probability calculation
- For prediction: Ignore the missing feature

### Q12: Can Naive Bayes be used for regression?
**Answer:** Not typically. Naive Bayes is primarily a classification algorithm. However, you can adapt it for regression by predicting the expected value using probability distributions.

### Q13: How does Gaussian Naive Bayes work with continuous features?
**Answer:** It assumes each class follows a normal distribution:
- For each class, estimate mean and variance of each feature
- Use Gaussian probability density function to calculate likelihood
- Apply Bayes' theorem for final prediction

### Q14: What is the difference between Multinomial and Bernoulli Naive Bayes?
**Answer:**
- **Multinomial:** Uses count data (e.g., word frequencies in documents)
- **Bernoulli:** Uses binary features (feature present or not)
- Bernoulli can work with features that are simply present/absent

### Q15: Why does Naive Bayes work well despite the independence assumption?
**Answer:** Even though the independence assumption is rarely true, Naive Bayes:
- Still ranks classes correctly (not exact probabilities)
- Is robust to irrelevant features
- Works well for classification even with correlated features
- The errors from wrong independence often cancel out

## Scenario-Based Questions

### Q16: Your Naive Bayes model has very low accuracy. What might be wrong?
**Answer:**
- Features are highly correlated
- Independence assumption doesn't hold
- Need feature engineering
- Consider using another classifier

### Q17: In spam detection, Naive Bayes assigns high probability to spam but marks legitimate email as spam. Why?
**Answer:** This is likely because:
- Certain words are too dominant
- Prior probability of spam is too high
- Need to tune threshold
- Consider using class weights
