# Bag of Words (BoW) - Exercises

## 🎯 Practice Problems

### Exercise 1: Build BoW from Scratch
**Problem:** Implement a simple Bag of Words vectorizer from scratch (without using sklearn).

**Input:** 
```python
documents = [
    "I love dogs",
    "I love cats",
    "I love both cats and dogs"
]
```

**Expected Output:**
- Vocabulary: sorted unique words
- BoW matrix as 2D array

---

### Exercise 2: Binary BoW
**Problem:** Modify the CountVectorizer to use binary=True and explain the difference from regular BoW.

---

### Exercise 3: Custom Vocabulary
**Problem:** Create a BoW vectorizer with a custom vocabulary that must include: ["python", "java", "code", "programming"]

**Test:** "I love programming in python"

---

### Exercise 4: Document Similarity
**Problem:** Given 3 documents, calculate pairwise cosine similarity and find the most similar pair.

```python
docs = [
    "Machine learning is great",
    "Deep learning is a subset of machine learning",
    "I love coding in python"
]
```

---

### Exercise 5: N-gram Analysis
**Problem:** Create BoW with bigrams and trigrams on this corpus:
```python
corpus = [
    "the cat sat on the mat",
    "the dog played with the cat"
]
```

List all unigrams, bigrams, and trigrams in the vocabulary.

---

### Exercise 6: Stopword Handling
**Problem:** Create a BoW model that:
1. Removes English stopwords
2. Keeps negation words (not, no, never)
3. Apply to: "I am not happy but the service was good"

---

### Exercise 7: Vocabulary Size Analysis
**Problem:** Analyze how vocabulary size affects model performance.

Given this corpus, find:
1. Total unique words
2. Words appearing in only 1 document
3. Words appearing in all documents
4. Suggest vocabulary limit

```python
corpus = [
    "the cat sat on the mat",
    "the dog ran to the mat",
    "the cat and the dog",
    "the mat is on the floor"
]
```

---

### Exercise 8: Sparse Matrix Conversion
**Problem:** Convert BoW vectors to dense array and calculate sparsity percentage.

```python
corpus = ["word " * i for i in range(1, 6)]  # Creates docs with increasing words
```

---

### Exercise 9: Text Classification Pipeline
**Problem:** Build a complete spam classifier using BoW + Naive Bayes.

**Data:**
```python
emails = [
    "Get free money now",           # spam
    "Meeting at 3pm",              # not spam
    "Click here to win prize",     # spam
    "Please review document",      # not spam
    "Congratulations winner",      # spam
    "Project deadline Friday"      # not spam
]
```

---

### Exercise 10: Inverse Document Frequency
**Problem:** Calculate IDF manually for each word in vocabulary.

**Formula:** IDF = log(N / df) where N = total docs, df = docs containing word

```python
corpus = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "cats are great pets"
]
```

---

## 🏆 Challenge Problems

### Challenge 1: Build Custom Vectorizer
Create a class `MyCountVectorizer` that mimics sklearn's CountVectorizer with:
- fit() method
- transform() method  
- fit_transform() method

### Challenge 2: Movie Review Sentiment Analysis
Build a sentiment classifier on movie reviews using BoW.

### Challenge 3: Document Search Engine
Create a simple search engine that:
- Indexes documents
- Ranks results by cosine similarity
- Handles new queries

---

## 📝 Answer Key

Solutions are in `solutions.py`
