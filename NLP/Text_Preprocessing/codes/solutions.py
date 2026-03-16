"""
Text Preprocessing - Exercise Solutions
=========================================
"""

import re
import string
from collections import Counter
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


# ==================== Exercise 1: Basic Tokenization ====================
def basic_tokenize(text):
    """
    Simple word tokenization without NLTK
    Removes punctuation and splits by whitespace
    """
    # Remove punctuation
    for punct in string.punctuation:
        text = text.replace(punct, "")

    # Split by whitespace and filter empty strings
    tokens = [word for word in text.split() if word]
    return tokens


# ==================== Exercise 2: Stemming Comparison ====================
def compare_stemmers(words):
    """Compare different stemmers on given words"""
    porter = PorterStemmer()

    # Note: For full comparison, you would use all three stemmers
    # This shows the pattern

    results = {}
    for word in words:
        results[word] = {
            "porter": porter.stem(word),
            # Add snowball and lancaster for full comparison
        }

    print("Word       | Porter")
    print("-" * 30)
    for word, stems in results.items():
        print(f"{word:10} | {stems['porter']}")

    return results


# ==================== Exercise 3: Custom Stopwords ====================
def custom_remove(text, custom_stopwords):
    """Remove custom stopwords from text"""
    words = text.split()
    filtered = [word for word in words if word not in custom_stopwords]
    return " ".join(filtered)


# ==================== Exercise 4: Text Cleaner Pipeline ====================
def clean_text(text):
    """Comprehensive text cleaning"""
    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # 3. Remove email addresses
    text = re.sub(r"\S+@\S+", "", text)

    # 4. Remove special characters (keep letters and numbers)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

    # 5. Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 6. Handle multiple exclamation marks
    text = re.sub(r"!+", "!", text)

    return text


# ==================== Exercise 5: Negation Handling ====================
def handle_negation(text):
    """Mark negation words and following words"""
    negation_words = {"not", "n't", "no", "never", "neither", "nobody", "nothing"}
    words = text.split()
    result = []

    for i, word in enumerate(words):
        if word.lower() in negation_words or word.lower().endswith("n't"):
            if i + 1 < len(words):
                result.append(f"NOT_{words[i + 1]}")
                words[i + 1] = ""  # Mark as handled
            else:
                result.append(word)
        elif words[i]:  # If not marked as handled
            result.append(word)

    return " ".join(result)


# ==================== Exercise 6: Lemmatization with POS ====================
def lemmatize_with_pos(words):
    """Lemmatize words using correct POS tags"""
    lemmatizer = WordNetLemmatizer()

    # Map POS tags to lemmatizer pos arguments
    pos_map = {
        "V": "v",  # verb
        "N": "n",  # noun
        "J": "a",  # adjective
        "R": "r",  # adverb
    }

    # Simple POS-based lemmatization (without actual POS tagging)
    # For production, use actual POS tagging
    results = []
    for word in words:
        # Default to verb lemmatization (most common)
        lemma = lemmatizer.lemmatize(word, "v")
        results.append(lemma)

    return results


# ==================== Exercise 7: Contraction Expander ====================
def expand_contractions(text):
    """Expand contractions in text"""
    contractions_map = {
        "n't": " not",
        "'re": " are",
        "'s": " is",
        "'d": " would",
        "'ll": " will",
        "'ve": " have",
        "'m": " am",
        "can't": "cannot",
        "won't": "will not",
        "don't": "do not",
        "doesn't": "does not",
        "didn't": "did not",
        "wouldn't": "would not",
        "couldn't": "could not",
        "shouldn't": "should not",
        "isn't": "is not",
        "aren't": "are not",
        "wasn't": "was not",
        "weren't": "were not",
        "haven't": "have not",
        "hasn't": "has not",
        "hadn't": "had not",
    }

    for contraction, expansion in contractions_map.items():
        text = re.sub(r"\b" + contraction + r"\b", expansion, text, flags=re.IGNORECASE)

    return text


# ==================== Exercise 8: Word Frequency After Preprocessing ====================
def preprocess_and_frequency(sentences):
    """Preprocess sentences and calculate word frequencies"""
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    all_words = []

    for sentence in sentences:
        # Tokenize
        tokens = word_tokenize(sentence.lower())

        # Remove stopwords
        tokens = [t for t in tokens if t not in stop_words and t.isalpha()]

        # Lemmatize
        tokens = [lemmatizer.lemmatize(t) for t in tokens]

        all_words.extend(tokens)

    return Counter(all_words)


# ==================== Exercise 9: Emoji Handler ====================
# Emoji to text mapping
EMOJI_MAP = {
    "😊": "happy",
    "😢": "sad",
    "🎉": "celebrate",
    "❤️": "love",
    "👍": "good",
    "👎": "bad",
    "😂": "funny",
    "😮": "surprised",
    "😢": "crying",
    "🔥": "hot",
    "💯": "perfect",
    "✨": "sparkle",
    "⭐": "star",
    "🙏": "pray",
    "💪": "strong",
}


def emoji_to_text(text):
    """Replace emojis with text descriptions"""
    result = text
    for emoji, word in EMOJI_MAP.items():
        result = result.replace(emoji, f" {word} ")
    return result


def count_emojis(text):
    """Count emojis in text"""
    count = 0
    for char in text:
        if char in EMOJI_MAP:
            count += 1
    return count


def remove_emojis(text):
    """Remove emojis from text"""
    result = text
    for emoji in EMOJI_MAP:
        result = result.replace(emoji, "")
    return result


# ==================== Exercise 10: Complete Preprocessing Pipeline ====================
class TextPreprocessor:
    """Complete text preprocessing pipeline"""

    def __init__(self, remove_stopwords=True, keep_negations=True):
        self.remove_stopwords = remove_stopwords
        self.keep_negations = keep_negations
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

        # Remove negation words if keeping them
        if keep_negations:
            self.stop_words.discard("not")
            self.stop_words.discard("no")
            self.stop_words.discard("n't")

        # Contractions map
        self.contractions = {
            "n't": " not",
            "'re": " are",
            "'s": " is",
            "'ll": " will",
            "'ve": " have",
            "'m": " am",
        }

    def expand_contractions(self, text):
        for contraction, expansion in self.contractions.items():
            text = text.replace(contraction, expansion)
        return text

    def preprocess(self, text):
        # 1. Expand contractions
        text = self.expand_contractions(text)

        # 2. Lowercase
        text = text.lower()

        # 3. Remove URLs
        text = re.sub(r"http\S+|www\.\S+", "", text)

        # 4. Remove emails
        text = re.sub(r"\S+@\S+", "", text)

        # 5. Remove special characters
        text = re.sub(r"[^a-zA-Z\s]", "", text)

        # 6. Tokenize
        tokens = word_tokenize(text)

        # 7. Remove stopwords
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words]

        # 8. Lemmatize
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]

        # 9. Return cleaned text
        return " ".join(tokens)


# ==================== Exercise 11: Named Entity Preservation ====================
def preprocess_for_ner(text):
    """
    Preprocess text while preserving named entities
    - Keep original case
    - Don't lemmatize entities
    """
    # Simple preprocessing that preserves entities
    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Don't lowercase - entities need case information!

    return text


# ==================== Exercise 12: Chinese Word Segmentation ====================
def max_match_segment(text, dictionary):
    """
    Maximum matching algorithm for Chinese word segmentation
    Forward maximum matching
    """
    # Simple dictionary-based approach
    result = []
    i = 0

    while i < len(text):
        matched = False
        # Try longest match first
        for j in range(len(text), i, -1):
            word = text[i:j]
            if word in dictionary:
                result.append(word)
                i = j
                matched = True
                break

        if not matched:
            # Single character if no match
            result.append(text[i])
            i += 1

    return result


# ==================== Testing ALL SOLUTIONS ====================
if __name__ == "__main__":
    print("=" * 50)
    print("Testing Exercise 1: Basic Tokenization")
    print("=" * 50)
    text = "Hello, World! NLP is amazing."
    print(f"Input: {text}")
    print(f"Output: {basic_tokenize(text)}")

    print("\n" + "=" * 50)
    print("Testing Exercise 3: Custom Stopwords")
    print("=" * 50)
    print(custom_remove("I love programming in Python", ["love", "in"]))

    print("\n" + "=" * 50)
    print("Testing Exercise 4: Text Cleaner")
    print("=" * 50)
    text = "Check https://example.com!!! Contact me at test@email.com #NLP #Python!!"
    print(f"Input: {text}")
    print(f"Output: {clean_text(text)}")

    print("\n" + "=" * 50)
    print("Testing Exercise 5: Negation Handling")
    print("=" * 50)
    text = "I am not happy and do not like this"
    print(f"Input: {text}")
    print(f"Output: {handle_negation(text)}")

    print("\n" + "=" * 50)
    print("Testing Exercise 7: Contraction Expander")
    print("=" * 50)
    text = "I'm going to the store. It's John's car. You shouldn't do that."
    print(f"Input: {text}")
    print(f"Output: {expand_contractions(text)}")

    print("\n" + "=" * 50)
    print("Testing Exercise 8: Word Frequency")
    print("=" * 50)
    sentences = [
        "The cat sat on the mat.",
        "The dog ran to the park.",
        "Cats and dogs are great pets.",
    ]
    print(f"Frequency: {preprocess_and_frequency(sentences)}")

    print("\n" + "=" * 50)
    print("Testing Exercise 9: Emoji Handler")
    print("=" * 50)
    text = "I love this! 😊 Great job! 🎉"
    print(f"Original: {text}")
    print(f"Text: {emoji_to_text(text)}")
    print(f"Count: {count_emojis(text)}")
    print(f"Cleaned: {remove_emojis(text)}")

    print("\n" + "=" * 50)
    print("Testing Exercise 10: Complete Pipeline")
    print("=" * 50)
    preprocessor = TextPreprocessor()
    text = "Don't forget to visit https://mysite.com for FREE gifts!!!"
    print(f"Input: {text}")
    print(f"Output: {preprocessor.preprocess(text)}")

    print("\n" + "=" * 50)
    print("Testing Exercise 11: NER Preprocessing")
    print("=" * 50)
    text = "Apple Inc. is headquartered in Cupertino, California."
    print(f"Input: {text}")
    print(f"Output: {preprocess_for_ner(text)}")
