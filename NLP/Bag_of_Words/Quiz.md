# Bag of Words (BoW) - Quiz

## 📝 Test Your Knowledge

### Question 1: Basic Concept
What does the Bag of Words model represent a document as?

A) A list of words in order
B) A set of unique words
C) A vector of word frequencies
D) A tree structure

**Answer: C** - BoW represents documents as vectors of word counts/frequencies.

---

### Question 2: Vocabulary Size
If you have a corpus with 1000 unique words, what is the dimension of each BoW vector?

A) 1
B) 1000
C) Depends on document length
D) Infinity

**Answer: B** - Each document is represented as a vector of length equal to the vocabulary size.

---

### Question 3: Word Order
Which statement is true about BoW?

A) It preserves word order
B) It ignores word order completely
C) It only keeps the first and last word
D) It reverses word order

**Answer: B** - BoW treats documents as "bags" of words, completely ignoring grammar and word order.

---

### Question 4: OOV Problem
What happens to words in test data that were not seen in training data?

A) They are automatically added to vocabulary
B) They are ignored
C) They cause errors
D) They are set to maximum value

**Answer: B** - Unknown/OOV words are typically ignored since they're not in the vocabulary.

---

### Question 5: Sparse Matrix
Why are BoW vectors considered "sparse"?

A) They contain mostly zeros
B) They contain mostly ones
C) They are compressed
D) They are encrypted

**Answer: A** - Most entries are zeros because each document only uses a small fraction of the vocabulary.

---

### Question 6: Binary BoW
What values does a binary BoW vector contain?

A) Only 0
B) Only 1
C) 0 or 1
D) Any integer

**Answer: C** - Binary BoW indicates only presence (1) or absence (0) of words.

---

### Question 7: n-grams
What bigram would be extracted from "machine learning"?

A) ["machine", "learning"]
B) ["machine learning"]
C) ["ma", "ch", "hi", "in", "ne", "el", "le", "ea", "ar", "rn", "rn", "ni", "in", "ng"]
D) ["learning machine"]

**Answer: B** - A bigram is a 2-word sequence. "machine learning" is one bigram.

---

### Question 8: Stopwords
Why are stopwords usually removed in BoW?

A) They carry important meaning
B) They appear in every document
C) They don't add discriminative value
D) They are too long

**Answer: C** - Stopwords (the, a, is) appear everywhere and don't help distinguish documents.

---

### Question 9: Cosine Similarity
When comparing BoW vectors, what does cosine similarity measure?

A) The length of vectors
B) The angle between vectors
C) The difference in word count
D) The vocabulary size

**Answer: B** - Cosine similarity measures the cosine of the angle between two vectors.

---

### Question 10: BoW vs TF-IDF
What does TF-IDF add to BoW?

A) Nothing
B) Word length normalization
C) Term importance weighting
D) Word order preservation

**Answer: C** - TF-IDF weights words by their importance (downweights common words, upweights rare distinctive words).

---

### Question 11: max_features
What does `max_features=1000` do in CountVectorizer?

A) Creates 1000 documents
B) Limits vocabulary to 1000 most common words
C) Creates 1000-dimensional vectors
D) Processes 1000 words per document

**Answer: B** - max_features limits the vocabulary to the top N most frequent terms.

---

### Question 12: Preprocessing
Which preprocessing step is NOT typically done before BoW?

A) Lowercasing
B) Removing punctuation
C) Keeping word order
D) Removing stopwords

**Answer: C** - Word order is not kept in BoW, so preprocessing focuses on cleaning the text.

---

### Question 13: Dimension Reduction
Which technique can reduce BoW dimensionality?

A) Word addition
B) PCA
C) Random amplification
D) None of the above

**Answer: B** - PCA (Principal Component Analysis) can reduce high-dimensional BoW vectors.

---

### Question 14: Spam Detection
What classifier works well with BoW for spam detection?

A) K-means
B) KNN
C) Naive Bayes
D) DBSCAN

**Answer: C** - Naive Bayes works well with BoW because it handles high-dimensional sparse data effectively and assumes feature independence like BoW.

---

### Question 15: Use Case
Which is NOT a good use case for BoW?

A) Sentiment classification
B) Language translation
C) Spam detection
D) Document clustering

**Answer: B** - Language translation requires understanding word order and context, which BoW cannot capture. BoW works for classification and clustering tasks.

---

## 🎯 Score Guide

| Score | Level |
|-------|-------|
| 13-15 | Expert 🥇 |
| 10-12 | Advanced 🥈 |
| 7-9 | Intermediate 🥉 |
| 4-6 | Beginner |
| 0-3 | Keep Learning! |

---

## 📚 Quick Review

Remember these key points:
- **BoW**: Text → word count vectors
- **Vocabulary**: All unique words from corpus
- **Vector Size**: Equal to vocabulary size
- **Order**: Lost completely
- **Sparsity**: Most values are zeros
- **OOV**: Unknown words ignored
- **n-grams**: Capture some context
- **Similarity**: Use cosine similarity
- **Improvement**: TF-IDF weights by importance
