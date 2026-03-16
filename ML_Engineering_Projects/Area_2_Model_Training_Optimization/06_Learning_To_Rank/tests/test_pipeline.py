import pytest
from src.pipeline import RankingPipeline, RankingResult


@pytest.fixture
def documents():
    return [
        "Python is a programming language used for machine learning",
        "The Eiffel Tower is located in Paris, France",
        "Deep learning uses neural networks with many layers",
        "Football is a popular sport played worldwide",
        "TensorFlow and PyTorch are ML frameworks",
    ]


def test_bm25_only_pipeline(documents):
    pipeline = RankingPipeline(documents, reranker=None, final_top_k=3)
    results = pipeline.search("machine learning frameworks")
    assert len(results) <= 3
    assert all(isinstance(r, RankingResult) for r in results)


def test_results_ranked(documents):
    pipeline = RankingPipeline(documents, reranker=None, final_top_k=5)
    results = pipeline.search("neural networks deep learning")
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_rank_assignment(documents):
    pipeline = RankingPipeline(documents, reranker=None, final_top_k=3)
    results = pipeline.search("Python programming")
    for i, r in enumerate(results, 1):
        assert r.rank == i


def test_ml_query_retrieves_ml_docs(documents):
    pipeline = RankingPipeline(documents, reranker=None, final_top_k=2)
    results = pipeline.search("machine learning deep learning")
    top_texts = [r.doc_text.lower() for r in results]
    assert any("learning" in t for t in top_texts)
