# TF-IDF - Quiz

## 📝 Test Your Knowledge

### Question 1: TF-IDF Definition
What does TF-IDF stand for?

A) Term Frequency-Inverse Document Frequency
B) Text Frequency-Inverse Data Frequency
C) Term Finding-Inverse Document Finder
D) Text Feature-Indexed Data File

**Answer: A** - Term Frequency-Inverse Document Frequency

---

### Question 2: IDF Calculation
If you have 100 documents and a word appears in 10 documents, what is the IDF? (Use log base e)

A) 10
B) 2.303
C) 0.1
D) 2

**Answer: B** - IDF = log(100/10) = log(10) ≈ 2.303

---

### Question 3: Term Frequency
What is the term frequency of "cat" in "The cat sat on the cat"?

A) 1
B) 2
C) 0.33
D) 0.5

**Answer: B** - "cat" appears 2 times out of 7 words

---

### Question 4: Stopwords
In TF-IDF, what happens to stopwords like "the" and "is"?

A) They get high TF-IDF scores
B) They get low TF-IDF scores
C) They are removed automatically
D) They are ignored in calculation

**Answer: B** - Since they appear in almost all documents, IDF is near zero, resulting in low TF-IDF scores.

---

### Question 5: TF-IDF vs BoW
Which is a key advantage of TF-IDF over simple Bag of Words?

A) Preserves word order
B) Weights words by importance
C) Works with images
D) Faster to compute

**Answer: B** - TF-IDF weights words based on their importance across the corpus.

---

### Question 6: OOV Problem
What happens if a test document contains words not in the training vocabulary?

A) Error occurs
B) They are ignored
C) They get maximum score
D) They replace common words

**Answer: B** - Unknown/OOV words are typically ignored in TF-IDF.

---

### Question 7: Cosine Similarity
What is typically used to measure document similarity with TF-IDF?

A) Euclidean distance
B) Manhattan distance
C) Cosine similarity
D) Jaccard similarity

**Answer: C** - Cosine similarity measures the angle between TF-IDF vectors.

---

### Question 8: Sublinear TF
What is sublinear TF?

A) Using log of term frequency
B) Using square root of TF
C) Using negative TF
D) Using TF without normalization

**Answer: A** - Sublinear TF uses 1 + log(TF) to diminish impact of high-frequency terms.

---

### Question 9: Sparsity
TF-IDF vectors are typically:

A) Dense
B) Sparse
C) Binary
D) Continuous

**Answer: B** - TF-IDF vectors are sparse since most words don't appear in most documents.

---

### Question 10: max_features
What does max_features=1000 do in TfidfVectorizer?

A) Creates 1000 documents
B) Limits vocabulary to 1000 most important terms
C) Processes 1000 words per document
D) Creates 1000-dimensional output

**Answer: B** - Limits vocabulary to top N features by term frequency across corpus.

---

### Question 11: Document Length
TF-IDF handles different document lengths by:

A) Ignoring length
B) Dividing by document length
C) Adding padding
D) Using fixed vocabulary

**Answer: B** - TF normalizes by document length, making comparison fair.

---

### Question 12: Use Case
Which is NOT a typical use case for TF-IDF?

A) Search engine ranking
B) Document classification
C) Image recognition
D) Keyword extraction

**Answer: C** - TF-IDF is for text, not images.

---

### Question 13: Vocabulary
What happens if vocabulary is too large?

A) Better representation
B) Overfitting, memory issues
C) Faster computation
D) More accurate

**Answer: B** - Large vocabulary leads to sparse vectors, overfitting, and memory issues.

---

### Question 14: Preprocessing
Which preprocessing is typically done before TF-IDF?

A) Adding more stopwords
B) Keeping all capital letters
C) Tokenization and lowercasing
D) Adding emoji

**Answer: C** - Standard preprocessing includes tokenization, lowercasing, and removing special characters.

---

### Question 15: Limitation
What is a key limitation of TF-IDF?

A) Cannot handle numbers
B) Ignores word order and semantics
C) Only works in English
D) Cannot handle large documents

**Answer: B** - TF-IDF doesn't capture word order or semantic meaning.

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

- **TF**: How often word appears in document
- **IDF**: How rare/distinctive word is across corpus
- **TF-IDF**: TF × IDF
- **High Score**: Rare but meaningful words
- **Low Score**: Common words (stopwords)
- **Use**: Search ranking, classification, keywords
- **Limitations**: No semantics, OOV, word independence
