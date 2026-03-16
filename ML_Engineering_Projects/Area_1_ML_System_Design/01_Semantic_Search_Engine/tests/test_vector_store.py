import numpy as np
import pytest

from src.config import SearchConfig
from src.vector_store import FAISSStore


@pytest.fixture
def faiss_store(tmp_path):
    config = SearchConfig(index_path=tmp_path / "test_index", vector_store="faiss")
    return FAISSStore(config)


def test_add_and_search(faiss_store):
    vecs = np.random.rand(5, 384).astype(np.float32)
    # Normalise
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    ids = [f"doc_{i}" for i in range(5)]
    texts = [f"text {i}" for i in range(5)]
    metadata = [{"source": "test"} for _ in range(5)]

    faiss_store.add(ids, vecs, texts, metadata)
    assert len(faiss_store) == 5

    query = vecs[0:1]
    results = faiss_store.search(query, k=3)
    assert len(results) == 3
    assert results[0].doc_id == "doc_0"  # should find itself as most similar


def test_empty_search_returns_empty(faiss_store):
    query = np.random.rand(1, 384).astype(np.float32)
    results = faiss_store.search(query, k=5)
    assert results == []


def test_save_and_load(faiss_store, tmp_path):
    vecs = np.random.rand(3, 384).astype(np.float32)
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    faiss_store.add(["a", "b", "c"], vecs, ["t1", "t2", "t3"], [{}, {}, {}])
    faiss_store.save()

    config = SearchConfig(index_path=tmp_path / "test_index", vector_store="faiss")
    store2 = FAISSStore(config)
    store2.load()
    assert len(store2) == 3
