from __future__ import annotations

from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """Neural cross-encoder reranker using sentence-transformers CrossEncoder.

    Cross-encoders process (query, document) pairs jointly through a transformer,
    enabling fine-grained interaction between query and document tokens.
    This is slower than bi-encoders but typically 5-10% more accurate on ranking tasks.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model = CrossEncoder(model_name, max_length=512)

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> list[tuple[float, str]]:
        """Score and rerank documents for a given query.

        Args:
            query: The search query.
            documents: List of candidate documents to rerank.
            top_k: Return only top-k results (None = return all).

        Returns:
            List of (score, document_text) sorted by descending score.
        """
        if not documents:
            return []
        pairs = [(query, doc) for doc in documents]
        scores = self.model.predict(pairs)
        ranked = sorted(
            zip(scores.tolist(), documents),
            key=lambda x: x[0],
            reverse=True,
        )
        return ranked[:top_k] if top_k else ranked

    def score(self, query: str, document: str) -> float:
        """Score a single (query, document) pair."""
        return float(self.model.predict([(query, document)])[0])
