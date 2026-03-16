import pytest
from src.config import RAGConfig
from src.chunker import Chunk
from src.retriever import ChromaRetriever


@pytest.fixture
def retriever(tmp_path):
    config = RAGConfig(
        chroma_persist_dir=tmp_path / "chroma",
        chroma_collection="test_col",
    )
    return ChromaRetriever(config)


def test_add_and_retrieve(retriever):
    chunks = [
        Chunk("Python is a programming language", {"source": "test", "chunk_index": 0}),
        Chunk("Machine learning optimizes model parameters", {"source": "test", "chunk_index": 1}),
        Chunk("The Eiffel Tower is in Paris", {"source": "test", "chunk_index": 2}),
    ]
    retriever.add_chunks(chunks)
    assert retriever.count() == 3

    results = retriever.retrieve("programming languages", k=2, use_mmr=False)
    assert len(results) > 0
    assert any("Python" in r.text for r in results)


def test_empty_retrieval_returns_empty(retriever):
    results = retriever.retrieve("anything", k=3)
    assert results == []


def test_retrieve_scores_in_range(retriever):
    chunks = [Chunk(f"text {i}", {"source": "t", "chunk_index": i}) for i in range(5)]
    retriever.add_chunks(chunks)
    results = retriever.retrieve("some query", k=3, use_mmr=False)
    for r in results:
        assert -1.0 <= r.score <= 1.1  # cosine similarity range
