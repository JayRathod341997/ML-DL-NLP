# Word Embeddings - Exercise Solutions

from gensim.models import Word2Vec

# Sample corpus
sentences = [
    ["king", "queen", "prince", "princess"],
    ["man", "woman", "boy", "girl"],
    ["dog", "cat", "pet", "animal"],
    ["king", "man", "royal"],
    ["queen", "woman", "royal"],
]

# Train Word2Vec
model = Word2Vec(sentences, vector_size=100, window=3, min_count=1)

# Get word vector
print("king vector:", model.wv["king"][:5])

# Similar words
print("Similar to king:", model.wv.most_similar("king", topn=3))

# Analogy: king - man + woman = queen
result = model.wv.most_similar(positive=["king", "woman"], negative=["man"])
print("king - man + woman =", result)
