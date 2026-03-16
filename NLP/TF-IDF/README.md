# TF-IDF (Term Frequency-Inverse Document Frequency)

## 📖 Explain Like I'm 5

Imagine you have a big book library!

You want to find which book is about what topic. 

Here's the trick:
- If a word appears A LOT in ONE book, that word is SPECIAL for that book! (like "dinosaurs" in a dinosaur book)
- If a word appears in ALMOST EVERY book (like "the", "is", "a"), it's NOT special at all!

So we calculate:
**TF** = How often a word appears in THIS book
**IDF** = How special is this word across ALL books

TF × IDF = How IMPORTANT is this word for THIS specific book!

Example:
- Word "the": High in every book (high TF), but not special (low IDF) → Low score
- Word "dinosaurs": Rare overall (low IDF), but high in dino book (high TF) → HIGH score!

This helps us find the MOST IMPORTANT words in any document!

## 🔍 What is TF-IDF?

TF-IDF is a numerical statistic that reflects how important a word is to a document in a collection or corpus.

### The Formula:

**TF (Term Frequency):**
```
TF(word, doc) = (Number of times word appears in doc) / (Total words in doc)
```

**IDF (Inverse Document Frequency):**
```
IDF(word) = log(Total documents / Documents containing word)
```

**TF-IDF:**
```
TF-IDF = TF × IDF
```

### Example:

Documents:
- D1: "cat sat on mat"
- D2: "dog sat on log"
- D3: "cat and dog"

Word "cat":
- TF in D1: 1/4 = 0.25
- TF in D2: 0/5 = 0
- TF in D3: 1/3 = 0.33

- IDF: log(3/2) = 0.405

- TF-IDF in D1: 0.25 × 0.405 = 0.10
- TF-IDF in D2: 0 × 0.405 = 0
- TF-IDF in D3: 0.33 × 0.405 = 0.13

## 💡 Where It Is Used?

### 1. **Search Engines**
- Google, Bing use TF-IDF variants
- Rank documents by query relevance
- Find keywords in documents

### 2. **Text Classification**
- Spam detection
- News categorization
- Sentiment analysis

### 3. **Document Summarization**
- Extract key sentences
- Keyword extraction
- Topic identification

### 4. **Information Retrieval**
- Find similar documents
- Recommendation systems
- Document clustering

### 5. **SEO**
- Analyze webpage content
- Keyword optimization
- Content relevance

## ⚙️ Benefits

1. **Weighted Term Importance**
   - Rare words get higher weights
   - Common words downweighted

2. **Handles Variable Document Lengths**
   - Normalizes by document length
   - Fair comparison across documents

3. **Simple and Fast**
   - Easy to implement
   - Computationally efficient

4. **Interpretable**
   - Scores explain keyword importance
   - Easy to debug

5. **Effective Baseline**
   - Works well for many tasks
   - Good starting point

## ⚠️ Limitations

1. **No Semantic Understanding**
   - Can't capture meaning
   - "Bank" (river) = "Bank" (money)

2. **No Context**
   - Ignores word order
   - Loses phrase meaning

3. **OOV Problem**
   - Can't handle new words
   - Vocabulary dependent

4. **Word Independence**
   - Doesn't capture relationships
   - "Hot dog" = "hot" + "dog"

5. **Length Bias**
   - Longer documents may have artificially lower scores

## 🏢 Enterprise Level Example

### Netflix - Content Recommendation

Netflix uses TF-IDF to find similar content:

1. **Process:**
   - Extract keywords from movie descriptions
   - Calculate TF-IDF for each title
   - Build similarity matrix

2. **Example:**
   - Movie A: "Exciting action adventure with explosions"
   - Movie B: "Action hero saves world"
   - "Action" gets high TF-IDF in both → Similar!

3. **Impact:**
   - 75% of Netflix viewing from recommendations
   - Saves $1B+ annually in customer retention

### LinkedIn - Job Matching

LinkedIn uses TF-IDF to match candidates with jobs:

1. **Process:**
   - Extract keywords from job descriptions
   - Match with candidate profiles
   - Rank by TF-IDF similarity

2. **Benefits:**
   - Automated matching
   - Improved candidate quality
   - Faster hiring process

## 📊 Implementation

### Using Scikit-learn:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    "This is the first document",
    "This document is the second document",
    "And this is the third one",
    "Is this the first document"
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

print(vectorizer.get_feature_names_out())
print(X.toarray())
```

### Key Parameters:

| Parameter | Description |
|-----------|-------------|
| `max_features` | Limit vocabulary size |
| `min_df` | Ignore rare terms |
| `max_df` | Ignore common terms |
| `ngram_range` | Include n-grams |
| `sublinear_tf` | Use log(TF) |

## 📊 Summary

| Aspect | Details |
|--------|---------|
| **Full Form** | Term Frequency-Inverse Document Frequency |
| **Purpose** | Weight words by importance |
| **Formula** | TF × IDF |
| **High Score** | Rare but important words |
| **Low Score** | Common words (stopwords) |

TF-IDF is a powerful technique for extracting meaningful keywords from documents. While modern methods like embeddings have surpassed it for semantic tasks, TF-IDF remains excellent for keyword extraction and as a baseline.

---

*Next: Learn about [Word Embeddings](../Word_Embeddings/README.md) - dense semantic representations.*
