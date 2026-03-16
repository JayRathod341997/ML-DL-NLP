# Topic Modeling - Solutions

from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Sample corpus
documents = [
    "machine learning is a subset of artificial intelligence",
    "deep learning uses neural networks",
    "python programming is essential for data science",
    "machine learning and deep learning are ai techniques",
    "data science involves programming and statistics",
]

# Vectorize
vectorizer = CountVectorizer(stop_words="english")
doc_term_matrix = vectorizer.fit_transform(documents)

# LDA
lda = LatentDirichletAllocation(n_components=2, random_state=42)
lda_output = lda.fit_transform(doc_term_matrix)

# Print topics
feature_names = vectorizer.get_feature_names_out()
for idx, topic in enumerate(lda.components_):
    top_words = [feature_names[i] for i in topic.argsort()[:-4:-1]]
    print(f"Topic {idx+1}: {', '.join(top_words)}")

# Document-topic distribution
print("\nDocument-Topic Distribution:")
for i, doc_topics in enumerate(lda_output):
    print(f"Doc {i+1}: {doc_topics}")
