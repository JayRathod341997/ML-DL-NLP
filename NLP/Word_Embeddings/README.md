# Word Embeddings

## 📖 Explain Like I'm 5

Imagine you have a magic map where similar words live close together!

Think of a playground:
- Dogs and cats are both pets → they stand near each other
- Apples and oranges are fruits → they stand together too
- Cars and bikes are vehicles → they're on another corner

This magic map is called "word embedding"! 

Computers can't understand words directly, so we give each word a list of numbers (like an address on our map). Words with similar meanings have similar addresses!

Example:
- "dog" → [0.8, 0.2, -0.5]
- "cat" → [0.75, 0.25, -0.45]
- "car" → [-0.3, 0.9, 0.1]

"dog" and "cat" are close! "car" is far away!

## 🔍 What are Word Embeddings?

Word embeddings are dense, low-dimensional vector representations of words that capture semantic meaning. Unlike sparse one-hot encoding (BoW), embeddings pack meaning into compact vectors.

### Types of Word Embeddings:

1. **Word2Vec** (Google, 2013)
   - Skip-gram: Predict context from word
   - CBOW: Predict word from context

2. **GloVe** (Stanford, 2014)
   - Global Vectors
   - Based on word co-occurrence statistics

3. **FastText** (Facebook, 2016)
   - Subword embeddings
   - Handles OOV better

4. **ELMO** (2018)
   - Contextual embeddings
   - Different for different contexts

5. **BERT** (2018+)
   - Deep contextual
   - Transformer-based

## 💡 Where It Is Used?

### 1. **Semantic Search**
- Find similar documents
- Question answering

### 2. **Text Classification**
- Sentiment analysis
- Spam detection

### 3. **Machine Translation**
- Map words between languages
- Preserve meaning

### 4. **Information Retrieval**
- Find relevant documents
- Recommendation systems

### 5. **NLP Downstream Tasks**
- NER, POS tagging
- Text summarization

## ⚙️ Benefits

1. **Dense Representations**
   - Smaller than one-hot
   - More meaningful

2. **Semantic Similarity**
   - Similar words cluster
   - Analogies work!

3. **Pre-trained Models**
   - Download and use
   - Transfer learning

4. **Generalization**
   - Handle synonyms
   - Out-of-vocabulary help

## ⚠️ Limitations

1. **Fixed Representations**
   - Polysemy issues
   - "Bank" = river or money?

2. **Requires Large Data**
   - Training from scratch needs lots of text

3. **Out-of-Vocabulary**
   - New words problematic
   - Subword helps but not perfect

4. **No Context**
   - (Except ELMO, BERT)
   - Same word, same embedding

## 🏢 Enterprise Example

### Google News - Article Clustering
- Embeddings cluster similar articles
- Personalized news feeds
- 70% improvement in click-through

### Amazon - Product Recommendations
- Product descriptions embedded
- Similar products suggested
- 35% of purchases from recommendations
