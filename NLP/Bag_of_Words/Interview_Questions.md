# Bag of Words - Interview Questions & Answers

## ❓ Frequently Asked Questions

### 1. What is Bag of Words (BoW) in NLP?

**Answer:**
Bag of Words is a text representation technique that converts text into numerical vectors by counting word frequencies, ignoring grammar and word order but keeping multiplicity.

**Key Steps:**
1. Create vocabulary from all documents
2. For each document, count occurrences of each vocabulary word
3. Represent document as a vector of these counts

**Example:**
```
Documents: 
- "I love dogs"
- "I love cats"

Vocabulary: [I, love, dogs, cats]

BoW Vectors:
- "I love dogs" → [1, 1, 1, 0]
- "I love cats" → [1, 1, 0, 1]
```

---

### 2. What are the limitations of Bag of Words?

**Answer:**

| Limitation | Explanation |
|------------|-------------|
| **Loss of Order** | "Dog bites man" = "Man bites dog" - same representation |
| **Vocabulary Size** | Large vocabularies create high-dimensional sparse vectors |
| **OOV Problem** | Unknown words in test data can't be represented |
| **No Semantics** | Can't capture meaning, synonyms, or context |
| **Feature Importance** | Common words dominate regardless of importance |
| **Sparsity** | Most values are zero, inefficient memory usage |

---

### 3. How does BoW differ from TF-IDF?

**Answer:**

| Aspect | BoW | TF-IDF |
|--------|-----|--------|
| **Definition** | Raw word counts | Term Frequency × Inverse Document Frequency |
| **Weighting** | Equal for all documents | Weights by importance across corpus |
| **Common Words** | High values (problem) | Low values (they're common) |
| **Rare Words** | Low values | High values (more distinctive) |
| **Use Case** | Simple tasks | Better for information retrieval |

**Key Difference:** TF-IDF downweights common words and upweights rare but important words.

---

### 4. What is the difference between CountVectorizer and TfidfVectorizer in scikit-learn?

**Answer:**

**CountVectorizer:**
- Creates bag of words (term frequency)
- Just counts occurrences
- `transform(corpus)` → term-document matrix

```python
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
```

**TfidfVectorizer:**
- Creates TF-IDF vectors
- Weights by term importance
- Similar API but uses TF-IDF weighting

```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
```

---

### 5. How do you handle out-of-vocabulary words in BoW?

**Answer:**

**Approaches:**

1. **Discard Unknown Words** (most common)
   - Words not in vocabulary are ignored
   - Information loss but simple

2. **Use <UNK> Token**
   - Replace all OOV words with <UNK>
   - Preserves information about OOV presence

3. **Character-level Features**
   - Add n-grams (bigrams, trigrams)
   - OOV words might share n-grams with known words

4. **Subword Tokenization**
   - Use BPE or WordPiece
   - Decompose OOV into known subwords

5. **Vocabulary Expansion**
   - Use word embeddings
   - Map OOV to nearest known word

---

### 6. What is n-grams in BoW? Why is it used?

**Answer:**

N-grams are contiguous sequences of n items (words, characters).

**Types:**
- **Unigrams**: Single words (1-gram) → "the", "cat"
- **Bigrams**: 2-word sequences → "the cat", "cats are"
- **Trigrams**: 3-word sequences → "the cat sat"

**Why Use N-grams:**

1. **Capture Context**: "not good" is different from "good"
2. **Phrase Detection**: "New York", "machine learning"
3. **Reduce Ambiguity**: "bank" (river vs money) - context helps
4. **Better Features**: More informative than single words

**Implementation:**
```python
from sklearn.feature_extraction.text import CountVectorizer

# Bigrams
vectorizer = CountVectorizer(ngram_range=(1, 2))

# Unigrams + Bigrams
vectorizer = CountVectorizer(ngram_range=(1, 2))
```

---

### 7. How do you decide vocabulary size for BoW?

**Answer:**

**Factors to Consider:**

1. **Dataset Size**
   - Small dataset → smaller vocabulary
   - Large dataset → can handle larger vocabulary

2. **Document Length**
   - Long documents → larger vocabulary needed
   - Short documents → smaller vocabulary sufficient

3. **max_features Parameter**
   - Limit to top N most frequent words
   - `CountVectorizer(max_features=10000)`

4. **min_df and max_df**
   - `min_df`: Ignore terms appearing in fewer than N documents
   - `max_df`: Ignore terms appearing in more than X% of documents
   - Removes rare and overly common words

**Best Practice:**
```python
vectorizer = CountVectorizer(
    max_features=5000,  # Keep top 5000
    min_df=2,           # Appear in at least 2 docs
    max_df=0.95        # Not in >95% of docs
)
```

---

### 8. What is sparse matrix representation? Why is it important for BoW?

**Answer:**

**Sparse Matrix:**
A matrix where most elements are zero. For BoW, typical documents use only a small fraction of the vocabulary.

**Example:**
```
Vocabulary: 10,000 words
Document: Uses ~50 words
Sparsity: 99.5% zeros!
```

**Why Important:**

1. **Memory Efficiency**: Don't store all zeros
2. **Computational Speed**: Only process non-zero elements
3. **Required for Large Scale**: BoW with full vocab would be impossible otherwise

**Scikit-learn Implementation:**
```python
# CSR (Compressed Sparse Row) format
from scipy import sparse

# Instead of dense numpy array
X_dense = np.zeros((1000, 10000))

# Use sparse matrix
X_sparse = sparse.csr_matrix(X_dense)
# Only stores non-zero values!
```

---

### 9. Can BoW be used for sentence similarity? How?

**Answer:**

**Yes! Using Cosine Similarity:**

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

corpus = [
    "I love dogs",
    "I love cats", 
    "Dogs are great"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

# Calculate similarities
similarity = cosine_similarity(X)

# Doc 0 vs Doc 1: "I love dogs" vs "I love cats" → 0.67
# Doc 0 vs Doc 2: "I love dogs" vs "Dogs are great" → 0.41
```

**Process:**
1. Convert sentences to BoW vectors
2. Use cosine similarity to measure angle between vectors
3. Higher similarity = more similar documents

**Limitation:** Still loses word order and context!

---

### 10. How do you preprocess text before applying BoW?

**Answer:**

**Standard Pipeline:**

1. **Lowercase**: "Hello" → "hello"
2. **Remove Punctuation**: "!@#$%^&*()"
3. **Tokenize**: Split into words
4. **Remove Stopwords**: the, a, is, are...
5. **Stemming/Lemmatization**: running → run
6. **Handle Contractions**: don't → do not

**Implementation:**
```python
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def preprocess(text):
    # Lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = text.split()
    
    # Remove stopwords
    stop = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    
    return ' '.join(tokens)
```

---

### 11. What is the difference between binary BoW and count BoW?

**Answer:**

| Aspect | Binary BoW | Count BoW |
|--------|------------|-----------|
| **Values** | 0 or 1 | 0, 1, 2, 3... |
| **Meaning** | Present/Absent | Frequency |
| **Use Case** | When presence matters | When frequency matters |
| **Sensitivity** | Less sensitive | More sensitive |

**Example:**
```
Document: "dog dog cat"

Binary BoW: [1, 0, 1]  # dog=1 (present), cat=1 (present)
Count BoW:   [2, 0, 1]  # dog appears 2 times
```

---

### 12. How would you use BoW for spam detection?

**Answer:**

**Step-by-Step:**

1. **Collect Data**: Ham + Spam emails
2. **Preprocess**: Clean text, tokenize, remove stopwords
3. **Create BoW**: Build vocabulary from training data
4. **Vectorize**: Convert emails to vectors
5. **Train Classifier**: Naive Bayes or Logistic Regression

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('vectorizer', CountVectorizer(max_features=5000)),
    ('classifier', MultinomialNB())
])

# Train
pipeline.fit(train_emails, train_labels)

# Predict
predictions = pipeline.predict(test_emails)
```

**Why Naive Bayes works well:**
- BoW creates discrete counts
- NB handles high-dimensional sparse data well
- Assumes word independence (like BoW!)

---

### 13. What is the "curse of dimensionality" in BoW?

**Answer:**

**Problem:**
As vocabulary size grows, the feature space becomes very high-dimensional, causing:

1. **Data Sparsity**: Each document uses tiny fraction of vocabulary
2. **Overfitting**: Models memorize rather than learn patterns
3. **Computational Cost**: More features = more computation
4. **Memory Issues**: Large sparse matrices

**Solutions:**

1. **Limit Vocabulary**: `max_features`
2. **Remove Rare Words**: `min_df`
3. **Remove Common Words**: `max_df`
4. **Dimensionality Reduction**: PCA, SVD
5. **Regularization**: L1/L2 in classifiers

---

### 14. Compare BoW with word embeddings.

**Answer:**

| Aspect | BoW | Word Embeddings |
|--------|-----|-----------------|
| **Dimensionality** | High (vocab size) | Low (100-300) |
| **Representation** | Sparse | Dense |
| **Context** | No | Yes |
| **Pre-training** | No | Yes |
| **OOV Handling** | Poor | Moderate |
| **Training Data** | Any | Large corpus needed |
| **Semantic** | No | Yes (somewhat) |

**Key Insight:** BoW captures frequency, embeddings capture meaning.

---

### 15. How would you build a document search engine using BoW?

**Answer:**

**Architecture:**

1. **Indexing Phase:**
   ```
   Documents → Preprocess → BoW Vectors → Store in Database
   ```

2. **Query Phase:**
   ```
   Query → Preprocess → BoW Vector → Compare with Index → Return Results
   ```

**Implementation:**
```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Index documents
corpus = [doc1, doc2, doc3, ...]
vectorizer = CountVectorizer(max_features=10000)
doc_vectors = vectorizer.fit_transform(corpus)

def search(query):
    # Convert query to vector
    query_vec = vectorizer.transform([query])
    
    # Calculate similarity
    similarities = cosine_similarity(query_vec, doc_vectors)
    
    # Return top matches
    top_indices = np.argsort(similarities[0])[-5:][::-1]
    return [corpus[i] for i in top_indices]
```

**Ranking:** Use cosine similarity scores to rank results.
