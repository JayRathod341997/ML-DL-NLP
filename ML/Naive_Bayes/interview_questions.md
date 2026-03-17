# Naive Bayes - Interview Questions & System Design

## Fundamentals

### What is Naive Bayes?

Probabilistic classifier based on Bayes' theorem with "naive" independence assumption between features.

### Bayes' Theorem

```
P(Class|Features) = P(Features|Class) × P(Class) / P(Features)
```

### Types of Naive Bayes

| Type | Use Case |
|------|----------|
| Gaussian | Continuous features |
| Multinomial | Text/count data |
| Bernoulli | Binary features |
| Categorical | Categorical features |

---

## Production Issues

### 1. Zero Probability

**Problem**: unseen feature values cause zero probability

**Solutions**:
- Laplace smoothing (add-one smoothing)
- Bayesian smoothing

### 2. Feature Independence

**Problem**: Naive assumption rarely true

**Solutions**:
- Feature selection
- Use domain knowledge
- Consider other algorithms

### 3. Underflow

**Problem**: Multiplying many small probabilities

**Solutions**:
- Use log probabilities
- Work in log space

---

## Short Q&A

| Question | Answer |
|----------|--------|
| **Why is it called "naive"?** | Assumes feature independence |
| **When is Naive Bayes good?** | Text classification, spam detection, quick baseline |
| **What is Laplace smoothing?** | Adding 1 to counts to avoid zero probabilities |
| **What is the time complexity?** | O(n × d × c) for training, O(d) for prediction |
| **Does it need feature scaling?** | No - uses probabilities |
| **What are its advantages?** | Fast, works with small data, handles many features |

---

## Follow-up Questions

### How would you apply Naive Bayes to text classification?

```
1. Text Preprocessing
   - Tokenization
   - Remove stopwords
   - Stemming/lemmatization

2. Feature Extraction
   - TF-IDF vectorization
   - Bag of words

3. Model Training
   - Use MultinomialNB
   - Apply Laplace smoothing

4. Optimization
   - Feature selection
   - Parameter tuning
```
