"""Ranking evaluation metrics: NDCG@k, MAP, MRR.

Pure numpy/Python implementations — no external dependencies.
"""

from __future__ import annotations

import numpy as np


def ndcg_at_k(relevance: list[int], k: int) -> float:
    """Normalised Discounted Cumulative Gain at rank k.

    Args:
        relevance: List of relevance labels in ranked order (index 0 = rank 1).
                   Labels can be binary (0/1) or graded (0/1/2/3).
        k: Cutoff rank.

    Returns:
        NDCG@k score in [0, 1].
    """
    relevance = relevance[:k]
    if not relevance:
        return 0.0
    dcg = sum(
        (2 ** rel - 1) / np.log2(rank + 2)
        for rank, rel in enumerate(relevance)
    )
    ideal = sorted(relevance, reverse=True)
    idcg = sum(
        (2 ** rel - 1) / np.log2(rank + 2)
        for rank, rel in enumerate(ideal)
    )
    return float(dcg / idcg) if idcg > 0 else 0.0


def average_precision(relevance: list[int]) -> float:
    """Average Precision for a single query.

    Args:
        relevance: Binary relevance list in ranked order.

    Returns:
        AP score in [0, 1].
    """
    if not any(relevance):
        return 0.0
    hits, total_ap = 0, 0.0
    for rank, rel in enumerate(relevance, 1):
        if rel:
            hits += 1
            total_ap += hits / rank
    num_relevant = sum(relevance)
    return total_ap / num_relevant if num_relevant else 0.0


def mean_average_precision(relevance_lists: list[list[int]]) -> float:
    """Mean Average Precision over multiple queries."""
    if not relevance_lists:
        return 0.0
    return float(np.mean([average_precision(r) for r in relevance_lists]))


def reciprocal_rank(relevance: list[int]) -> float:
    """Reciprocal Rank: 1/rank of first relevant document."""
    for rank, rel in enumerate(relevance, 1):
        if rel:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(relevance_lists: list[list[int]]) -> float:
    """Mean Reciprocal Rank over multiple queries."""
    if not relevance_lists:
        return 0.0
    return float(np.mean([reciprocal_rank(r) for r in relevance_lists]))


def evaluate_ranking(
    all_relevance: list[list[int]],
    k: int = 10,
) -> dict[str, float]:
    """Compute all ranking metrics for a set of queries.

    Args:
        all_relevance: List of relevance lists, one per query.
        k: Cutoff for NDCG@k.

    Returns:
        dict with ndcg_at_k, map, mrr.
    """
    ndcg = float(np.mean([ndcg_at_k(r, k) for r in all_relevance]))
    map_score = mean_average_precision(all_relevance)
    mrr = mean_reciprocal_rank(all_relevance)
    return {
        f"ndcg@{k}": round(ndcg, 4),
        "map": round(map_score, 4),
        "mrr": round(mrr, 4),
    }
