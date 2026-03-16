# Text Preprocessing - Quiz

## 📝 Test Your Knowledge

### Question 1: Tokenization
What is the output of tokenizing "NLP is fun!" using word tokenization?

A) ["NLP", "is", "fun"]
B) ["NLP", "is", "fun", "!"]
C) ["NLP", "is", "fun", "!"]
D) ["NLP", "is", "fun", "!"]
E) ["NLP is fun!"]

**Answer: B** - Most word tokenizers split on whitespace but keep punctuation as separate tokens.

---

### Question 2: Stemming vs Lemmatization
Which technique produces a valid dictionary word?

A) Stemming only
B) Lemmatization only
C) Both
D) Neither

**Answer: B** - Lemmatization always produces valid dictionary words (lemmas), while stemming may not.

---

### Question 3: Stopwords
Why might keeping stopwords be important for sentiment analysis?

A) They reduce vocabulary size
B) They speed up processing
C) Negation words like "not" change sentiment entirely
D) They carry no meaning

**Answer: C** - "not good" vs "good" have opposite meanings; removing "not" breaks sentiment analysis.

---

### Question 4: Text Normalization
Which of the following is NOT typically part of text normalization?

A) Lowercasing
B) Removing special characters
C) Tokenization
D) All of the above are part of normalization

**Answer: D** - All are part of text normalization.

---

### Question 5: Out-of-Vocabulary (OOV)
Which approach best handles OOV words?

A) Using a larger vocabulary
B) Subword tokenization (BPE, WordPiece)
C) Removing all unknown words
D) Converting to lowercase

**Answer: B** - Subword tokenization can represent OOV words using known subword units.

---

### Question 6: Chinese Tokenization
Why is Chinese tokenization more challenging than English?

A) Chinese has more words
B) Chinese doesn't use spaces between words
C) Chinese uses a different alphabet
D) Chinese is harder to spell

**Answer: B** - Chinese text doesn't have word boundaries marked by spaces, requiring word segmentation.

---

### Question 7: Contractions
How should "won't" be handled for semantic analysis?

A) Remove it completely
B) Keep as is
C) Expand to "will not"
D) Convert to "wont"

**Answer: C** - Expanding contractions preserves the semantic meaning (will + not).

---

### Question 8: POS Tagging
What does POS tagging stand for?

A) Part of Speech
B) Point of Sale
C) Position of Sentence
D) Parts of System

**Answer: A** - POS tagging = Part-of-Speech tagging (noun, verb, adjective, etc.)

---

### Question 9: Emoji Handling
What is the best approach for emojis in sentiment analysis?

A) Always remove them
B) Always keep them
C) Convert to text representation or keep as features
D) Replace with random words

**Answer: C** - Emojis carry significant sentiment information and should be handled thoughtfully.

---

### Question 10: NER Preprocessing
For Named Entity Recognition, should you lowercase all text?

A) Yes, always
B) No, casing helps identify entities
C) Only for proper nouns
D) Only for common nouns

**Answer: B** - Preserving casing helps models identify entities like "Apple" (company) vs "apple" (fruit).

---

### Question 11: Library Choice
Which library is best for production-grade text preprocessing?

A) NLTK only
B) spaCy
C) TextBlob
D) Regular expressions only

**Answer: B** - spaCy is designed for production with fast, accurate processing and modern architecture.

---

### Question 12: Lemmatization vs Stemming Speed
Which is generally faster?

A) Lemmatization
B) Stemming
C) They are the same speed
D) Depends on the implementation

**Answer: B** - Stemming uses simple rule-based chopping, while lemmatization requires dictionary lookups.

---

### Question 13: Multi-language Text
What is the first step when preprocessing multi-language text?

A) Translation
B) Language detection
C) Tokenization
D) Remove all non-English

**Answer: B** - Identifying the language allows using language-specific preprocessing pipelines.

---

### Question 14: Spelling Correction
Which method uses edit distance to find corrections?

A) Dictionary lookup
B) Levenshtein distance
C) Language model
D) Neural network only

**Answer: B** - Levenshtein distance measures minimum edits needed to transform one word to another.

---

### Question 15: Subword Tokenization
Why is BPE (Byte Pair Encoding) popular?

A) It's the oldest method
B) It reduces vocabulary size while handling OOV
C) It always produces real words
D) It doesn't need training

**Answer: B** - BPE creates a compact vocabulary and can represent unseen words through subword combinations.

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
- **Tokenization**: Breaking text into words/sentences/subwords
- **Stemming**: Fast, rule-based root finding (may not be real word)
- **Lemmatization**: Dictionary-based root (always real word)
- **Stopwords**: Common words, remove for most tasks BUT keep for sentiment
- **OOV**: Use subword tokenization to handle unseen words
- **Casing**: Preserve for NER, convert for classification
- **Negations**: Critical for sentiment - don't lose them!
