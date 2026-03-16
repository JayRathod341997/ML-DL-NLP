from __future__ import annotations

from dataclasses import dataclass

from rank_bm25 import BM25Okapi


@dataclass
class RankingResult:
    doc_text: str
    score: float
    rank: int


class RankingPipeline:
    """Two-stage ranking: BM25 retrieval → reranking.

    Stage 1 (BM25): fast keyword retrieval, high recall, top-100 candidates.
    Stage 2 (reranker): slower but more accurate cross-encoder or LambdaMART.
    """

    def __init__(
        self,
        documents: list[str],
        reranker=None,  # CrossEncoderReranker or LambdaMARTModel
        bm25_top_k: int = 100,
        final_top_k: int = 10,
    ) -> None:
        self.documents = documents
        self.reranker = reranker
        self.bm25_top_k = bm25_top_k
        self.final_top_k = final_top_k

        tokenised = [doc.lower().split() for doc in documents]
        self._bm25 = BM25Okapi(tokenised)

    def search(self, query: str) -> list[RankingResult]:
        """Search documents for a query using BM25 + optional reranking.

        Args:
            query: Natural language search query.

        Returns:
            List of RankingResult sorted by score (descending).
        """
        # Stage 1: BM25 retrieval
        tokenised_query = query.lower().split()
        bm25_scores = self._bm25.get_scores(tokenised_query)
        top_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True,
        )[: self.bm25_top_k]
        candidates = [(self.documents[i], bm25_scores[i]) for i in top_indices]

        # Stage 2: Rerank (if reranker provided)
        if self.reranker is not None:
            candidate_texts = [c[0] for c in candidates]
            reranked = self.reranker.rerank(query, candidate_texts, top_k=self.final_top_k)
            return [
                RankingResult(doc_text=doc, score=score, rank=i + 1)
                for i, (score, doc) in enumerate(reranked)
            ]

        # No reranker — return BM25 results
        return [
            RankingResult(doc_text=doc, score=score, rank=i + 1)
            for i, (doc, score) in enumerate(candidates[: self.final_top_k])
        ]
