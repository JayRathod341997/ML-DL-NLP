import numpy as np
import pytest

from src.config import SearchConfig
from src.embedder import SentenceEmbedder


@pytest.fixture
def embedder():
    config = SearchConfig(embed_model="sentence-transformers/all-MiniLM-L6-v2")
    return SentenceEmbedder(config)


def test_encode_returns_correct_shape(embedder):
    texts = ["hello world", "machine learning is great"]
    vecs = embedder.encode(texts)
    assert vecs.shape == (2, embedder.embedding_dim)


def test_encode_returns_float32(embedder):
    vecs = embedder.encode(["test"])
    assert vecs.dtype == np.float32


def test_encode_query_shape(embedder):
    vec = embedder.encode_query("what is deep learning?")
    assert vec.shape == (1, embedder.embedding_dim)


def test_normalised_embeddings(embedder):
    vecs = embedder.encode(["normalised vector test"])
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
