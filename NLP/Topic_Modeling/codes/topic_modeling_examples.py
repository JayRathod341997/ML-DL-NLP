# Topic Modeling Examples using sklearn

from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np

# Sample documents
documents = [
    "Machine learning is a subset of artificial intelligence",
    "Deep learning uses neural networks for learning",
    "Python programming is essential for data science",
    "Machine learning and deep learning are AI techniques",
    "Data science involves programming and statistics",
    "Neural networks are used in deep learning applications"
]

# Create document-term matrix
vectorizer = CountVectorizer(stop_words='english')
doc_term_matrix = vectorizer.fit_transform(documents)

# LDA Topic Modeling
lda_model = LatentDirichletAllocation(
    n_components=2,  # Number of topics
    random_state=42,
    max_iter=10
)

lda_output = lda_model.fit_transform(doc_term_matrix)

# Print topics
feature_names = vectorizer.get_feature_names_out()

print("LDA Topics:")
for topic_idx, topic in enumerate(lda_model.components_):
    print(f"\nTopic {topic_idx + 1}:")
    top_words = [feature_names[i] for i in topic.argsort()[:-5:-1]]
    print("  ", ", ".join(top_words))

# NMF Topic Modeling
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

nmf_model = NMF(n_components=2, random_state=42)
nmf_output = nmf_model.fit_transform(tfidf_matrix)

print("\n\nNMF Topics:")
for topic_idx, topic in enumerate(nmf_model.components_):
    print(f"\nTopic {topic_idx + 1}:")
    feature_names = tfidf_vectorizer.get_feature_names_out()
    top_words = [feature_names[i] for i in topic.argsort()[:-5:-1]]
    print("  ", ", ".join(top_words))
