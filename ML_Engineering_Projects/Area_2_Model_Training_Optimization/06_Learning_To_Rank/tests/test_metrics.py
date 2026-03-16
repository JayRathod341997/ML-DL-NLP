import pytest
from src.metrics import ndcg_at_k, average_precision, mean_reciprocal_rank, evaluate_ranking


def test_ndcg_perfect_ranking():
    # Perfect ranking: most relevant first
    relevance = [3, 2, 1, 0]
    assert ndcg_at_k(relevance, k=4) == pytest.approx(1.0)


def test_ndcg_empty_relevance():
    assert ndcg_at_k([], k=5) == 0.0


def test_ndcg_no_relevant():
    assert ndcg_at_k([0, 0, 0], k=3) == 0.0


def test_ndcg_single_relevant():
    relevance = [0, 0, 1]
    score = ndcg_at_k(relevance, k=3)
    assert 0.0 < score < 1.0  # not 0, not perfect


def test_average_precision_perfect():
    assert average_precision([1, 1, 0, 0]) == pytest.approx(1.0)


def test_average_precision_no_relevant():
    assert average_precision([0, 0, 0]) == 0.0


def test_average_precision_last_relevant():
    ap = average_precision([0, 0, 0, 1])
    assert ap == pytest.approx(1 / 4)


def test_mrr():
    assert mean_reciprocal_rank([[0, 1, 0]]) == pytest.approx(1 / 2)
    assert mean_reciprocal_rank([[1, 0, 0]]) == pytest.approx(1.0)
    assert mean_reciprocal_rank([[0, 0, 0]]) == 0.0


def test_evaluate_ranking_keys():
    result = evaluate_ranking([[1, 0, 1], [0, 1, 0]], k=3)
    assert "ndcg@3" in result
    assert "map" in result
    assert "mrr" in result
