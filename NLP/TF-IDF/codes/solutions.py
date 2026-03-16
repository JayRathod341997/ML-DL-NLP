"""
TF-IDF - Exercise Solutions
===========================
"""

import numpy as np
import math
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==================== Exercise 1: Manual TF-IDF ====================
def manual_tfidf():
    """Calculate TF-IDF manually"""
    corpus = ["the cat sat", "the dog sat", "cat and dog"]

    # TF for "cat" in doc 0
    tf = 1 / 3  # 1 occurrence out of 3 words

    # IDF for "cat"
    docs_with_cat = 2  # appears in doc 0 and 2
    N = 3
    idf = math.log(N / docs_with_cat)

    tfidf = tf * idf

    print(f"TF for 'cat' in doc 0: {tf}")
    print(f"IDF for 'cat': {idf:.4f}")
    print(f"TF-IDF: {tfidf:.4f}")

    return tfidf


# ==================== Exercise 2: TF-IDF from Scratch ====================
def tfidf_from_scratch(corpus):
    """Implement TF-IDF without sklearn"""

    # Get vocabulary
    vocab = sorted(set(" ".join(corpus).split()))

    N = len(corpus)

    # Calculate IDF for each word
    idf = {}
    for word in vocab:
        df = sum(1 for doc in corpus if word in doc.split())
        idf[word] = math.log(N / df)

    # Calculate TF-IDF for each document
    vectors = []
    for doc in corpus:
        words = doc.split()
        tf = {word: words.count(word) / len(words) for word in vocab}
        tfidf_vector = [tf[w] * idf[w] for w in vocab]
        vectors.append(tfidf_vector)

    return vocab, np.array(vectors)


# ==================== Exercise 3: TF-IDF vs BoW ====================
def compare_tfidf_bow():
    """Compare TF-IDF and BoW rankings"""

    corpus = [
        "the cat sat on the mat",
        "the dog sat on the log",
        "cat and dog are great",
    ]

    # BoW
    bow_vec = CountVectorizer()
    X_bow = bow_vec.fit_transform(corpus).toarray()

    # TF-IDF
    tfidf_vec = TfidfVectorizer()
    X_tfidf = tfidf_vec.fit_transform(corpus).toarray()

    print("Vocabulary:", bow_vec.get_feature_names_out())
    print("\nBoW Matrix:")
    print(X_bow)
    print("\nTF-IDF Matrix:")
    print(np.round(X_tfidf, 3))

    return X_bow, X_tfidf


# ==================== Exercise 4: Document Search ====================
def document_search():
    """Search engine using TF-IDF"""

    documents = [
        "Python is a great programming language",
        "Java is also popular for web development",
        "Machine learning uses Python and TensorFlow",
        "Web development with JavaScript and HTML",
    ]

    query = "Python programming language"

    # Vectorize
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(documents + [query])

    # Calculate similarity
    doc_vectors = X[:-1]
    query_vector = X[-1]

    similarities = cosine_similarity(query_vector, doc_vectors)[0]

    # Rank results
    results = sorted(zip(documents, similarities), key=lambda x: x[1], reverse=True)

    print("Query:", query)
    print("\nResults:")
    for doc, sim in results:
        print(f"  Score: {sim:.3f} - {doc}")

    return results


# ==================== Exercise 5: Keyword Extraction ====================
def extract_keywords():
    """Extract top keywords using TF-IDF"""

    doc = "Machine learning is a subset of artificial intelligence that enables systems to learn from data"

    vectorizer = TfidfVectorizer()
    vectorizer.fit([doc])

    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = vectorizer.transform([doc]).toarray()[0]

    # Sort by score
    keywords = sorted(
        zip(feature_names, tfidf_scores), key=lambda x: x[1], reverse=True
    )

    print("Document:", doc)
    print("\nTop Keywords:")
    for word, score in keywords[:5]:
        print(f"  {word}: {score:.4f}")

    return keywords[:5]


# ==================== TEST ALL ====================
if __name__ == "__main__":
    print("=" * 50)
    print("Exercise 1: Manual TF-IDF")
    print("=" * 50)
    manual_tfidf()

    print("\n" + "=" * 50)
    print("Exercise 2: TF-IDF from Scratch")
    print("=" * 50)
    corpus = ["the cat sat", "the dog sat", "cat and dog"]
    vocab, vectors = tfidf_from_scratch(corpus)
    print(f"Vocabulary: {vocab}")
    print(f"Vectors:\n{vectors}")

    print("\n" + "=" * 50)
    print("Exercise 3: TF-IDF vs BoW")
    print("=" * 50)
    compare_tfidf_bow()

    print("\n" + "=" * 50)
    print("Exercise 4: Document Search")
    print("=" * 50)
    document_search()

    print("\n" + "=" * 50)
    print("Exercise 5: Keyword Extraction")
    print("=" * 50)
    extract_keywords()
