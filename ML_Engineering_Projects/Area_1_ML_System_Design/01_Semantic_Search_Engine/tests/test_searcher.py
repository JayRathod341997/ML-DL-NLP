import numpy as np
import pytest

from src.config import SearchConfig
from src.indexer import Document, DocumentIndexer
from src.searcher import SemanticSearcher


@pytest.fixture
def indexed_searcher(tmp_path):
    config = SearchConfig(
        index_path=tmp_path / "test_index",
        vector_store="faiss",
        top_k=3,
    )
    docs = [
        Document("1", "Python is a programming language", {"source": "test"}),
        Document("2", "Machine learning uses algorithms to learn from data", {"source": "test"}),
        Document("3", "Neural networks are inspired by the human brain", {"source": "test"}),
        Document("4", "The capital of France is Paris", {"source": "test"}),
        Document("5", "Deep learning is a subset of machine learning", {"source": "test"}),
    ]
    indexer = DocumentIndexer(config)
    indexer.index(docs)
    indexer.save()

    searcher = SemanticSearcher(config)
    return searcher


def test_search_returns_results(indexed_searcher):
    results = indexed_searcher.search("what is deep learning?")
    assert len(results) > 0


def test_search_results_sorted_by_score(indexed_searcher):
    results = indexed_searcher.search("machine learning algorithms")
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_search_top_k_respected(indexed_searcher):
    results = indexed_searcher.search("programming", k=2)
    assert len(results) <= 2


def test_search_ml_query_returns_ml_docs(indexed_searcher):
    results = indexed_searcher.search("deep learning neural networks", k=3)
    top_texts = [r.text.lower() for r in results]
    assert any("learning" in t or "neural" in t for t in top_texts)
