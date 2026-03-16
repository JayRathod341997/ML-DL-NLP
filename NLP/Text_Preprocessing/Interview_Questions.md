# Text Preprocessing - Interview Questions & Answers

## ❓ Frequently Asked Questions

### 1. What is the difference between stemming and lemmatization?

**Answer:**
Both aim to reduce words to their root form, but differ in approach:

| Aspect | Stemming | Lemmatization |
|--------|----------|---------------|
| **Method** | Chops off word endings | Uses dictionary/grammar rules |
| **Speed** | Faster | Slower |
| **Accuracy** | Less accurate | More accurate |
| **Example** | "running" → "run" | "better" → "good" |
| **Example** | "studies" → "studi" | "went" → "go" |

**Key Difference**: Stemming doesn't guarantee a real word, while lemmatization always produces a valid word.

---

### 2. Why do we remove stopwords? Are there any cases where we shouldn't remove them?

**Answer:**
Stopwords (the, is, a, an, and, but) appear frequently but carry little semantic meaning.

**Why Remove:**
- Reduces vocabulary size
- Improves processing speed
- Focuses on meaningful words
- Better for topic modeling, classification

**When NOT to Remove:**
- **Sentiment Analysis**: "not good" → removing "not" changes meaning
- **Machine Translation**: Stopwords needed for grammatical correctness
- **Question Answering**: Question words (who, what, where) are important
- **Spam Detection**: "You" or "FREE" patterns matter

---

### 3. What is tokenization? Name different types of tokenization.

**Answer:**
Tokenization is breaking text into smaller units (tokens) - words, sentences, or subwords.

**Types:**
1. **Word Tokenization**: Split by spaces
   - "I love NLP" → ["I", "love", "NLP"]

2. **Sentence Tokenization**: Split by sentence endings
   - "Hello. How are you?" → ["Hello.", "How are you?"]

3. **Subword Tokenization**: Break into meaningful subwords
   - "unhappiness" → ["un", "happiness"]
   - Used in BPE, WordPiece, SentencePiece

4. **Character Tokenization**: Each character as token
   - "cat" → ["c", "a", "t"]

---

### 4. What is the problem with simple tokenization for languages like Chinese?

**Answer:**
Chinese doesn't use spaces between words.

**Example:**
```
English: "I love natural language processing"
Chinese: "我爱自然语言处理" (no spaces!)
```

**Solution:**
- **Word Segmentation**: Identify word boundaries
- "我爱自然语言处理" → ["我", "爱", "自然语言处理"] or ["我", "爱", "自然", "语言", "处理"]
- Uses algorithms like Maximum Matching, CRF, or neural models

---

### 5. How would you handle misspellings in text data?

**Answer:**
Multiple approaches:

1. **Dictionary-based**: Use libraries like `pyspellchecker`
   ```python
   from spellchecker import SpellChecker
   spell = SpellChecker()
   corrected = spell.correction("ths")
   # Returns: "this"
   ```

2. **Levenshtein Distance**: Find closest word in dictionary
   ```python
   # Minimum edits to transform one word to another
   "teh" → "the" (1 substitution)
   ```

3. **Probabilistic Models**: Use language models to predict correct spelling

4. **Phonetic Algorithms**: Soundex, Metaphone for pronunciation-based matching

5. **Neural Networks**: Seq2Seq models for spelling correction

---

### 6. What is text normalization? Give examples.

**Answer:**
Text normalization converts text to a consistent, standard format.

**Common Techniques:**
1. **Lowercasing**: "Hello WORLD" → "hello world"
2. **Removing Special Characters**: "Hello!@#" → "Hello"
3. **Handling Contractions**: "can't" → "cannot", "won't" → "will not"
4. **Removing Numbers**: "Hello123" → "Hello" (or keep if meaningful)
5. **URL/Email Removal**: Remove or replace with tokens
6. **HTML Tag Removal**: Strip HTML from web content
7. **Unicode Normalization**: Convert different encodings to standard form

---

### 7. Explain the challenge of handling emojis and emoticons in text preprocessing.

**Answer:**
Emojis carry significant meaning, especially in social media.

**Challenges:**
- Thousands of different emojis
- Different platforms render differently
- Combinations (emoji + modifier) create more variations
- Emoticons :) :( :D have meanings

**Approaches:**
1. **Remove**: Simplest but loses sentiment
2. **Replace with text**: "😊" → "happy_face"
3. **Keep as features**: Use for sentiment analysis
4. **Emoji Embeddings**: Learn emoji representations

---

### 8. What is the difference between NLTK, spaCy, and TextBlob for text preprocessing?

**Answer:**

| Library | Best For | Features |
|---------|----------|----------|
| **NLTK** | Education, research | Many algorithms, steep learning curve |
| **spaCy** | Production, speed | Fast, modern, pretrained models |
| **TextBlob** | Simple tasks | Easy API, good for beginners |

**Code Comparison:**
```python
# NLTK
import nltk
nltk.word_tokenize("Hello world")
nltk.pos_tag(tokens)

# spaCy
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("Hello world")

# TextBlob
from textblob import TextBlob
blob = TextBlob("Hello world")
blob.words
blob.tags
```

---

### 9. How do you handle out-of-vocabulary (OOV) words?

**Answer:**
OOV words are unseen during training but appear in test data.

**Solutions:**
1. **Subword Tokenization**: BPE, WordPiece
   - "unseen" → ["un", "seen"]

2. **Character-level Models**: Process character by character

3. **Backoff**: Use <UNK> token for unknown words

4. **FastText**: Use subword embeddings
   - Word vector = sum of subword vectors

5. **BPE Dropout**: Randomly drop subword merges during training

---

### 10. What is the importance of preserving or removing punctuation?

**Answer:**

**Keep Punctuation When:**
- Sentiment analysis ("Great!!!" vs "Great.")
- Question classification (ending with "?")
- Detecting excitement/emotion
- Processing formal documents

**Remove Punctuation When:**
- Building vocabulary for classification
- Focus on semantic content
- Reducing vocabulary size
- Topic modeling

---

### 11. How would you preprocess text for a sentiment analysis model?

**Answer:**
Pipeline for sentiment analysis:

```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def preprocess_for_sentiment(text):
    # Lowercase
    text = text.lower()
    
    # Handle negations (important for sentiment!)
    text = re.sub(r"n't", " not", text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Keep emoticons as features
    # Don't remove exclamation marks - they indicate sentiment!
    
    # Tokenize
    tokens = text.split()
    
    # Remove stopwords EXCEPT negation words
    stop_words = set(stopwords.words('english'))
    stop_words.remove('not')
    stop_words.remove("n't")
    tokens = [t for t in tokens if t not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    
    return tokens
```

---

### 12. What is the difference between word-level and subword-level tokenization?

**Answer:**

| Aspect | Word-level | Subword-level |
|--------|------------|---------------|
| **Vocab Size** | Large (100K+) | Smaller (30K-50K) |
| **OOV Handling** | Can't handle | Can decompose |
| **Example** | "unseen" → UNK | "un" + "seen" |
| **Speed** | Faster | Slightly slower |
| **Quality** | Better for large vocab | Better for morphologically rich languages |

**Subword Algorithms:**
- BPE (Byte Pair Encoding)
- WordPiece
- SentencePiece

---

### 13. How do you handle multi-language text preprocessing?

**Answer:**
Challenges:
- Different tokenization rules
- Different stopword lists
- Different morphological structures

**Approaches:**
1. **Language Detection**: Identify language first
2. **Language-specific pipelines**: Process each language separately
3. **Language-agnostic**: Use character-level or subword
4. **Multilingual models**: Use mBERT, XLM-R

**Tools:**
- `langdetect` for language detection
- `spaCy` has multilingual models
- `fastText` supports 157 languages

---

### 14. What is the difference between hard tokenization and soft tokenization?

**Answer:**

**Hard Tokenization:**
- Fixed rules, no ambiguity
- Split by specific delimiter (space, punctuation)
- Example: "don't" → ["don", "'", "t"]

**Soft Tokenization:**
- Learned from data
- Uses statistical methods
- Example: "don't" → ["don't"] or ["do", "n't"] based on data

**Modern Approach:**
- Use pre-trained tokenizers (BERT uses WordPiece)
- They learn from large corpora
- Handle edge cases better

---

### 15. How would you preprocess text for Named Entity Recognition (NER)?

**Answer:**
NER requires careful preprocessing to preserve entity information:

```python
def preprocess_for_ner(text):
    # Don't lowercase - preserve proper nouns!
    # Keep punctuation that marks sentence boundaries
    
    # Handle special characters but preserve entity structure
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    
    # Don't remove numbers within entities (e.g., "Company2020")
    # Don't lemmatize - entities should stay as-is
    
    # Keep original casing for entity recognition
    # SpaCy models use casing as feature
    
    return text.strip()
```

**Key Rules:**
- Preserve original casing
- Don't remove numbers in entities
- Keep punctuation for sentence boundaries
- Don't stem or lemmatize entities
