# Word Embeddings - Interview Questions

## Q1: What are word embeddings?

Dense, low-dimensional vector representations of words that capture semantic meaning. Unlike one-hot vectors, embeddings pack meaning into compact representations.

## Q2: Difference between Word2Vec Skip-gram and CBOW?

- **Skip-gram**: Predict context words from target word (good for small datasets)
- **CBOW**: Predict target word from context words (faster, better for frequent words)

## Q3: What is the curse of dimensionality?

As vocabulary grows, one-hot vectors become huge and sparse. Embeddings solve this by using fixed-size dense vectors (typically 100-300 dimensions).

## Q4: How do embeddings capture analogies?

King - Man + Woman = Queen
This works because embeddings encode semantic relationships as vector arithmetic.

## Q5: What is GloVe?

Global Vectors - combines global word co-occurrence statistics with local context window methods.

## Q6: Handle out-of-vocabulary words?

- FastText: Use subword embeddings
- Character-level models
- Map to <UNK> token

## Q7: Difference between static and contextual embeddings?

- Static: Same embedding for word regardless of context
- Contextual: Different embedding based on surrounding context (ELMO, BERT)
