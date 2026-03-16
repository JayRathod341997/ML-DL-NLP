# TF-IDF - Interview Questions & Answers

## ❓ Frequently Asked Questions

### 1. What is TF-IDF and why is it used?

**Answer:**
TF-IDF (Term Frequency-Inverse Document Frequency) is a weighting scheme used in information retrieval and text mining to evaluate how important a word is to a document in a collection.

**Formula:**
```
TF-IDF(word, doc) = TF(word, doc) × IDF(word)
```

Where:
- TF = Term Frequency (how often word appears in document)
- IDF = log(Total Documents / Documents containing word)

**Why Use:**
- Distinguishes important words from common words
- Handles variable document lengths
- Ranks documents by relevance

---

### 2. Explain TF (Term Frequency) and IDF (Inverse Document Frequency)

**Answer:**

**Term Frequency (TF):**
- Measures how frequently a term occurs in a document
- Since documents vary in length, normalize by document length
- TF = (Count of word in document) / (Total words in document)
- Alternative: Just raw count

**Inverse Document Frequency (IDF):**
- Measures how rare/common a word is across all documents
- IDF = log(Total Documents / Documents with term)
- Words appearing in all documents get low IDF (not distinctive)
- Rare words get high IDF (distinctive/important)

**Example:**
```
Corpus: 1000 documents
Word "the": appears in 1000 documents → IDF = log(1000/1000) = 0
Word "dinosaur": appears in 5 documents → IDF = log(1000/5) = 5.3
```

---

### 3. How does TF-IDF differ from simple word counts (BoW)?

**Answer:**

| Aspect | BoW | TF-IDF |
|--------|-----|--------|
| Weighting | Equal weights | Weighted by importance |
| "the" | High count | Near zero |
| "unique term" | Low count | High |
| Document length | Not normalized | Normalized |
| Use case | Basic counting | Relevance ranking |

**Key Difference:** TF-IDF downweights common words and upweights distinctive words.

---

### 4. What are the limitations of TF-IDF?

**Answer:**

1. **No Semantic Understanding**
   - Can't understand meaning
   - "Bank" = river bank or money bank

2. **Ignores Word Order**
   - Loses context
   - "dog bites man" = "man bites dog"

3. **Vocabulary Limitations**
   - OOV words can't be represented
   - Requires known vocabulary

4. **Word Independence**
   - No relationships between words
   - "hot dog" not captured

5. **Length Bias**
   - Longer documents may have lower scores
   - Sublinear TF helps but not complete solution

---

### 5. How do you handle the OOV (Out-of-Vocabulary) problem in TF-IDF?

**Answer:**

1. **Use <UNK> Token**
   - Replace unknown words with generic token

2. **Character N-grams**
   - Use character-level features
   - Partial matching possible

3. **Subword Tokenization**
   - BPE, WordPiece
   - Decompose unknown words

4. **Vocabulary Expansion**
   - Word embeddings
   - Map to nearest known word

5. **Smoothing**
   - Add small constant to avoid division by zero

---

### 6. What is sublinear TF? When would you use it?

**Answer:**

**Sublinear TF:**
```
TF = 1 + log(count)
```

Instead of raw count: 1 + log(10) = 2.0 instead of 10

**Why Use:**
- Diminishes impact of very frequent terms
- Prevents one word from dominating
- More balanced weighting
- Used when you have documents with varying frequencies

```python
TfidfVectorizer(sublinear_tf=True)
```

---

### 7. How would you use TF-IDF for document similarity?

**Answer:**

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

docs = ["doc1 text", "doc2 text", "doc3 text"]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(docs)

# Cosine similarity
similarity = cosine_similarity(X[0], X[1])
```

**Steps:**
1. Convert all documents to TF-IDF vectors
2. Use cosine similarity (angle between vectors)
3. Higher similarity = more related documents

---

### 8. What is the difference between TfidfVectorizer and CountVectorizer?

**Answer:**

**CountVectorizer:**
- Produces term frequency counts
- Raw word counts: [2, 0, 1, ...]
- Basic bag of words

**TfidfVectorizer:**
- Applies TF-IDF weighting
- Considers document frequency
- Better for relevance ranking

```python
# Count
CountVectorizer().fit_transform(corpus)  # [[2, 0, 1]]

# TF-IDF  
TfidfVectorizer().fit_transform(corpus)  # [[0.5, 0, 0.3]]
```

---

### 9. How do you select optimal max_features for TF-IDF?

**Answer:**

**Methods:**

1. **Grid Search**
   ```python
   for max_features in [1000, 5000, 10000]:
       # test performance
   ```

2. **Based on Corpus Size**
   - Small corpus: smaller vocabulary
   - Large corpus: can handle larger vocab

3. **min_df and max_df**
   - `min_df=2`: Remove words in < 2 docs (rare)
   - `max_df=0.95`: Remove words in > 95% docs (too common)

4. **Domain Knowledge**
   - Keep domain-specific important terms

---

### 10. Can TF-IDF be used for multi-class classification? How?

**Answer:**

**Yes!**

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Vectorize
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(texts)

# Multi-class classifier
clf = LogisticRegression(multi_class='multinomial')
clf.fit(X, labels)

# Predict
predictions = clf.predict(new_text)
```

Works with:
- Logistic Regression
- Naive Bayes
- SVM
- Neural Networks

---

### 11. What is L1 vs L2 normalization in TF-IDF?

**Answer:**

**L1 Normalization (Manhattan):**
- Sum of absolute values = 1
- Each document has equal weight
- Sparse but not unit length

**L2 Normalization (Euclidean):**
- Square root of sum of squares = 1
- Standard in TF-IDF
- Better for cosine similarity

```python
TfidfVectorizer(norm='l1')  # or 'l2'
```

---

### 12. How would you extract keywords from a document using TF-IDF?

**Answer:**

```python
from sklearn.feature_extraction.text import TfidfVectorizer

doc = "your document text here"

vectorizer = TfidfVectorizer()
vectorizer.fit([doc])  # Single document

# Get feature names and scores
feature_names = vectorizer.get_feature_names_out()
tfidf_scores = vectorizer.transform([doc]).toarray()[0]

# Sort by score
keywords = sorted(zip(feature_names, tfidf_scores), key=lambda x: x[1], reverse=True)

print(keywords[:10])  # Top 10 keywords
```

---

### 13. What is the difference between TF-IDF and BM25?

**Answer:**

| Aspect | TF-IDF | BM25 |
|--------|--------|------|
| **Formula** | TF × IDF | More complex |
| **Saturation** | No | Yes (caps high TF) |
| **Length Norm** | Simple | Sophisticated |
| **Use Case** | General | Search engines |
| **Parameters** | None | k1, b |

**BM25 (Best Matching 25):**
- Used by search engines (Elasticsearch)
- Better handles document length
- Saturates term frequency
- More tunable parameters

---

### 14. How do you preprocess text for TF-IDF?

**Answer:**

**Standard Pipeline:**

1. Lowercase
2. Remove special characters
3. Tokenize
4. Remove stopwords
5. Stem/Lemmatize
6. Apply TF-IDF

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words='english',
    ngram_range=(1, 2),
    max_features=5000
)
```

---

### 15. When should you NOT use TF-IDF?

**Answer:**

1. **Semantic Understanding Needed**
   - Understanding meaning/context
   - Use: Word embeddings, Transformers

2. **Short Texts**
   - Tweets, titles
   - Limited vocabulary to work with

3. **Code/Mathematical Text**
   - Not regular natural language

4. **Multilingual with Different Structures**
   - Complex morphologies

5. **When Word Order Matters**
   - Use: RNNs, Transformers
