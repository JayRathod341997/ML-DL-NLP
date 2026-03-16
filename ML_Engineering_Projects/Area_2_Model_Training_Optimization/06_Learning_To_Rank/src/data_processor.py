from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RankingExample:
    query_id: str
    query: str
    doc_id: str
    text: str
    relevance: int  # 0 = not relevant, 1 = relevant


def load_msmarco_examples(
    max_queries: int = 5000,
    split: str = "train",
) -> list[RankingExample]:
    """Load MS MARCO passage ranking examples.

    Each query gets a set of passages with binary relevance labels.
    """
    from datasets import load_dataset

    ds = load_dataset("ms_marco", "v1.1", split=split, streaming=True)
    examples = []
    query_count = 0

    for row in ds:
        qid = str(query_count)
        query = row["query"]
        passages = row["passages"]
        texts = passages["passage_text"]
        is_selected = passages["is_selected"]

        for i, (text, rel) in enumerate(zip(texts, is_selected)):
            examples.append(
                RankingExample(
                    query_id=qid,
                    query=query,
                    doc_id=f"{qid}_doc_{i}",
                    text=text,
                    relevance=int(rel),
                )
            )
        query_count += 1
        if query_count >= max_queries:
            break

    return examples


def examples_to_feature_matrix(
    examples: list[RankingExample],
    bm25_scores: dict[tuple[str, str], float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert ranking examples to LightGBM feature matrix.

    Features per (query, doc) pair:
    - bm25_score
    - query_length
    - doc_length
    - query_term_overlap (fraction of query terms in doc)

    Returns:
        (X, y, groups) where groups = number of docs per query.
    """
    features, labels, group_sizes = [], [], []
    current_qid, count = None, 0

    for ex in examples:
        q_terms = set(ex.query.lower().split())
        d_terms = ex.text.lower().split()
        overlap = len(q_terms & set(d_terms)) / max(len(q_terms), 1)
        bm25 = bm25_scores.get((ex.query_id, ex.doc_id), 0.0)

        features.append([
            bm25,
            len(ex.query.split()),
            len(d_terms),
            overlap,
        ])
        labels.append(ex.relevance)

        if ex.query_id != current_qid:
            if current_qid is not None:
                group_sizes.append(count)
            current_qid = ex.query_id
            count = 0
        count += 1

    if count:
        group_sizes.append(count)

    return np.array(features, dtype=np.float32), np.array(labels), np.array(group_sizes)
