# Text Preprocessing - Exercises

## 🎯 Practice Problems

### Exercise 1: Basic Tokenization
**Problem:** Write a function that performs word tokenization on a given text without using NLTK. Split by spaces and handle punctuation.

**Input:** `"Hello, World! NLP is amazing."`

**Expected Output:** `["Hello", "World", "NLP", "is", "amazing"]`

---

### Exercise 2: Stemming Comparison
**Problem:** Compare Porter, Snowball, and Lancaster stemmers on the following words:
`["running", "children", "better", "university", "friendship"]`

Print which stemmers produce the same output and which differ.

---

### Exercise 3: Custom Stopwords
**Problem:** Create a function that removes custom stopwords from a text. The function should accept both a text and a list of custom stopwords to remove.

**Example:**
```python
custom_remove("I love programming in Python", ["love", "in"])
# Output: "I programming Python"
```

---

### Exercise 4: Text Cleaner Pipeline
**Problem:** Create a comprehensive text cleaning function that:
1. Converts to lowercase
2. Removes URLs
3. Removes email addresses
4. Removes special characters (keep only letters and numbers)
5. Removes extra whitespace
6. Handles multiple exclamation marks (replace with single)

**Input:** `"Check https://example.com!!! Contact me at test@email.com #NLP #Python!!"`

**Expected:** `"check contact me at #nlp #python"`

---

### Exercise 5: Negation Handling
**Problem:** Write a function that marks negation words and their following words. This is crucial for sentiment analysis.

**Input:** `"I am not happy and do not like this"`

**Expected Output:** `"I am NOT_happy and do NOT_like this"`

---

### Exercise 6: Lemmatization with POS
**Problem:** Write a function that uses the correct POS tag to lemmatize words properly. Test it on:
`["running", "better", "ate", "studies", "children", "went"]`

---

### Exercise 7: Contraction Expander
**Problem:** Create a dictionary of common contractions and expand them in text.

**Contractions to handle:** `'t`, `'s`, `'re`, `'ll`, `'ve`, `'m`, `'d`

**Input:** `"I'm going to the store. It's John's car. You shouldn't do that."`

---

### Exercise 8: Word Frequency After Preprocessing
**Problem:** Given a list of sentences:
```python
sentences = [
    "The cat sat on the mat.",
    "The dog ran to the park.",
    "Cats and dogs are great pets."
]
```

Perform full preprocessing (tokenize, remove stopwords, lemmatize) and calculate word frequencies.

---

### Exercise 9: Emoji Handler
**Problem:** Create functions to:
1. Replace emojis with text descriptions
2. Count emojis in text
3. Remove emojis

**Input:** `"I love this! 😊 Great job! 🎉"`

---

### Exercise 10: Complete Preprocessing Pipeline
**Problem:** Build a complete preprocessing pipeline class that:
1. Handles contractions
2. Lowercases text
3. Removes URLs and emails
4. Tokenizes
5. Removes stopwords
6. Lemmatizes
7. Returns cleaned text (joined)

**Test with:** `"Don't forget to visit https://mysite.com for FREE gifts!!!"`

---

### Exercise 11: Named Entity Preservation
**Problem:** For NER tasks, text should NOT be lowercased. Create a preprocessing function that:
1. Tokenizes preserving original case
2. Keeps proper nouns intact
3. Only normalizes non-entity tokens

**Input:** `"Apple Inc. is headquartered in Cupertino, California."`

---

### Exercise 12: Chinese Word Segmentation
**Problem:** Since Chinese doesn't use spaces, explain how you would segment:
`"我喜欢自然语言处理"` (I like natural language processing)

Using a simple maximum matching approach (dictionary-based), how would you segment it?

---

## 🏆 Challenge Problems

### Challenge 1: Build Your Own Tokenizer
Implement a tokenizer from scratch that:
1. Handles contractions
2. Keeps punctuation separate
3. Handles numbers (keep as one token)
4. Handles special characters

### Challenge 2: Spelling Corrector
Implement a simple spell checker using:
1. Dictionary lookup
2. Levenshtein distance for suggestions

### Challenge 3: Multi-language Preprocessor
Create a preprocessing pipeline that:
1. Detects language
2. Applies language-specific stopwords
3. Handles language-specific tokenization

---

## 📝 Answer Key

Solutions are provided in `solutions.py` file.
