"""
Bag of Words - Exercise Solutions
=================================
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
import math


# ==================== Exercise 1: Build BoW from Scratch ====================
def build_bow_from_scratch(documents):
    """Build BoW from scratch without sklearn"""

    # Get all unique words
    vocabulary = sorted(set(" ".join(documents).split()))

    # Create vectors
    vectors = []
    for doc in documents:
        doc_words = doc.split()
        vector = [doc_words.count(word) for word in vocabulary]
        vectors.append(vector)

    return vocabulary, np.array(vectors)


# ==================== Exercise 2: Binary BoW ====================
def binary_bow_example():
    """Demonstrate binary vs count BoW"""

    corpus = ["dog dog cat", "cat cat cat"]

    # Count BoW
    count_vec = CountVectorizer(binary=False)
    X_count = count_vec.fit_transform(corpus).toarray()

    # Binary BoW
    binary_vec = CountVectorizer(binary=True)
    X_binary = binary_vec.fit_transform(corpus).toarray()

    print("Count BoW:")
    print(X_count)
    print("\nBinary BoW:")
    print(X_binary)

    return X_count, X_binary


# ==================== Exercise 3: Custom Vocabulary ====================
def custom_vocabulary_example():
    """Use custom vocabulary"""

    custom_vocab = ["python", "java", "code", "programming"]

    vectorizer = CountVectorizer(vocabulary=custom_vocab)

    text = "I love programming in python"
    X = vectorizer.fit_transform([text]).toarray()

    print(f"Text: {text}")
    print(f"Vocabulary: {vectorizer.get_feature_names_out()}")
    print(f"Vector: {X[0]}")

    return X


# ==================== Exercise 4: Document Similarity ====================
def document_similarity():
    """Calculate cosine similarity between documents"""

    docs = [
        "Machine learning is great",
        "Deep learning is a subset of machine learning",
        "I love coding in python",
    ]

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(docs).toarray()

    # Calculate similarities
    similarities = cosine_similarity(X)

    print("Cosine Similarity Matrix:")
    print(np.round(similarities, 3))

    # Find most similar pair
    max_sim = 0
    pair = (0, 0)
    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            if similarities[i][j] > max_sim:
                max_sim = similarities[i][j]
                pair = (i, j)

    print(f"\nMost similar pair: Doc {pair[0]} and Doc {pair[1]}")
    print(f"Similarity: {max_sim:.3f}")
    print(f"Doc {pair[0]}: {docs[pair[0]]}")
    print(f"Doc {pair[1]}: {docs[pair[1]]}")

    return similarities


# ==================== Exercise 5: N-gram Analysis ====================
def ngram_analysis():
    """Analyze different n-grams"""

    corpus = ["the cat sat on the mat", "the dog played with the cat"]

    # Unigrams
    uni_vec = CountVectorizer(ngram_range=(1, 1))
    X_uni = uni_vec.fit_transform(corpus)
    print("Unigrams:", uni_vec.get_feature_names_out())

    # Bigrams
    bi_vec = CountVectorizer(ngram_range=(2, 2))
    X_bi = bi_vec.fit_transform(corpus)
    print("Bigrams:", bi_vec.get_feature_names_out())

    # Trigrams
    tri_vec = CountVectorizer(ngram_range=(3, 3))
    X_tri = tri_vec.fit_transform(corpus)
    print("Trigrams:", tri_vec.get_feature_names_out())

    return uni_vec, bi_vec, tri_vec


# ==================== Exercise 6: Stopword Handling ====================
def stopword_handling():
    """Handle stopwords with negation preservation"""

    # Custom stopwords (remove common but keep negation)
    stop_words = set(
        [
            "i",
            "me",
            "my",
            "myself",
            "am",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "but",
            "or",
            "and",
            "the",
            "a",
            "an",
            "in",
            "on",
            "at",
            "to",
            "for",
        ]
    )

    text = "I am not happy but the service was good"
    tokens = text.lower().split()

    # Filter
    filtered = [t for t in tokens if t not in stop_words or t in ["not", "no", "never"]]

    print(f"Original: {text}")
    print(f"After filtering: {' '.join(filtered)}")

    return filtered


# ==================== Exercise 7: Vocabulary Size Analysis ====================
def vocabulary_analysis():
    """Analyze vocabulary size and distribution"""

    corpus = [
        "the cat sat on the mat",
        "the dog ran to the mat",
        "the cat and the dog",
        "the mat is on the floor",
    ]

    # Get all words
    all_words = " ".join(corpus).split()
    unique_words = set(all_words)

    # Count per document
    doc_words = [set(doc.split()) for doc in corpus]

    # Words in only 1 document
    once = [w for w in unique_words if sum(1 for d in doc_words if w in d) == 1]

    # Words in all documents
    all_docs = [
        w for w in unique_words if sum(1 for d in doc_words if w in d) == len(corpus)
    ]

    print(f"Total unique words: {len(unique_words)}")
    print(f"Words in only 1 doc: {once}")
    print(f"Words in all docs: {all_docs}")
    print(f"\nSuggested max_features: {len(unique_words) - len(once)}")

    return len(unique_words), once, all_docs


# ==================== Exercise 8: Sparse Matrix Analysis ====================
def sparse_analysis():
    """Analyze sparsity of BoW matrix"""

    corpus = ["word " * i for i in range(1, 6)]
    corpus = [c.strip() for c in corpus]

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)

    nnz = X.nnz
    total = X.shape[0] * X.shape[1]
    sparsity = (1 - nnz / total) * 100

    print(f"Matrix shape: {X.shape}")
    print(f"Non-zero elements: {nnz}")
    print(f"Sparsity: {sparsity:.2f}%")

    return sparsity


# ==================== Exercise 9: Spam Classifier ====================
def spam_classifier():
    """Build spam classifier with BoW"""

    emails = [
        "Get free money now",
        "Meeting at 3pm",
        "Click here to win prize",
        "Please review document",
        "Congratulations winner",
        "Project deadline Friday",
    ]

    labels = [1, 0, 1, 0, 1, 0]

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(emails)

    clf = MultinomialNB()
    clf.fit(X, labels)

    # Test
    test_emails = ["Free money award", "Team meeting"]
    X_test = vectorizer.transform(test_emails)
    predictions = clf.predict(X_test)

    print("Training data:")
    for email, label in zip(emails, labels):
        print(f"  {email} -> {'spam' if label else 'not spam'}")

    print("\nPredictions:")
    for email, pred in zip(test_emails, predictions):
        print(f"  {email} -> {'spam' if pred else 'not spam'}")

    return clf, vectorizer


# ==================== Exercise 10: Calculate IDF ====================
def calculate_idf():
    """Calculate IDF manually"""

    corpus = ["the cat sat on the mat", "the dog sat on the log", "cats are great pets"]

    # Get vocabulary
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    vocab = vectorizer.get_feature_names_out()

    N = len(corpus)

    print(f"Vocabulary: {vocab}")
    print(f"N (total docs): {N}")
    print("\nIDF values:")

    for word in vocab:
        # Count docs containing this word
        df = sum(1 for doc in corpus if word in doc)
        idf = math.log(N / df)
        print(f"  {word}: df={df}, IDF={idf:.3f}")

    return vocab


# ==================== TEST ALL ====================
if __name__ == "__main__":
    print("=" * 50)
    print("Exercise 1: Build BoW from Scratch")
    print("=" * 50)
    docs = ["I love dogs", "I love cats", "I love both cats and dogs"]
    vocab, vectors = build_bow_from_scratch(docs)
    print(f"Vocabulary: {vocab}")
    print(f"Vectors:\n{vectors}")

    print("\n" + "=" * 50)
    print("Exercise 2: Binary vs Count BoW")
    print("=" * 50)
    binary_bow_example()

    print("\n" + "=" * 50)
    print("Exercise 3: Custom Vocabulary")
    print("=" * 50)
    custom_vocabulary_example()

    print("\n" + "=" * 50)
    print("Exercise 4: Document Similarity")
    print("=" * 50)
    document_similarity()

    print("\n" + "=" * 50)
    print("Exercise 5: N-gram Analysis")
    print("=" * 50)
    ngram_analysis()

    print("\n" + "=" * 50)
    print("Exercise 6: Stopword Handling")
    print("=" * 50)
    stopword_handling()

    print("\n" + "=" * 50)
    print("Exercise 7: Vocabulary Analysis")
    print("=" * 50)
    vocabulary_analysis()

    print("\n" + "=" * 50)
    print("Exercise 8: Sparsity Analysis")
    print("=" * 50)
    sparse_analysis()

    print("\n" + "=" * 50)
    print("Exercise 9: Spam Classifier")
    print("=" * 50)
    spam_classifier()

    print("\n" + "=" * 50)
    print("Exercise 10: IDF Calculation")
    print("=" * 50)
    calculate_idf()
