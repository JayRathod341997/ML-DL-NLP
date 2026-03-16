# Topic Modeling - Interview Questions

## Q1: What is Topic Modeling?

Topic modeling is an unsupervised learning technique that discovers abstract "topics" in a collection of documents. Each topic is a distribution over words, and each document is a distribution over topics.

## Q2: What is LDA?

Latent Dirichlet Allocation (LDA) is the most popular topic modeling algorithm. It assumes:
- Each document is a mixture of topics
- Each topic is a mixture of words

## Q3: Difference between LDA and NMF?

| Aspect | LDA | NMF |
|--------|-----|-----|
| Type | Probabilistic | Matrix Factorization |
| Output | Topic distributions | Non-negative components |
| Speed | Slower | Faster |

## Q4: How do you determine the number of topics?

1. **Domain Knowledge**: Based on expected number of topics
2. **Coherence Score**: Measure topic quality
3. **Grid Search**: Try different numbers and evaluate

## Q5: Applications of Topic Modeling?

1. Document organization
2. Content recommendation
3. Market research analysis
4. News categorization
5. Customer feedback analysis
