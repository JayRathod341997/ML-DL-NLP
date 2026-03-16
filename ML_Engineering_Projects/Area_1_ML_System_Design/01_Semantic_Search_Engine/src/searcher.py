from __future__ import annotations

from .config import SearchConfig
from .embedder import SentenceEmbedder
from .vector_store import SearchResult, VectorStore, get_vector_store


class SemanticSearcher:
    """Loads a persisted index and answers natural language queries."""

    def __init__(self, config: SearchConfig | None = None) -> None:
        self.config = config or SearchConfig()
        self.embedder = SentenceEmbedder(self.config)
        self.store: VectorStore = get_vector_store(self.config)
        self.store.load()

    def search(self, query: str, k: int | None = None) -> list[SearchResult]:
        """Search the index for the top-k documents most similar to the query.

        Args:
            query: Natural language query string.
            k: Number of results to return (defaults to config.top_k).

        Returns:
            List of SearchResult sorted by descending similarity score.
        """
        k = k or self.config.top_k
        query_vec = self.embedder.encode_query(query)
        results = self.store.search(query_vector=query_vec, k=k)
        return sorted(results, key=lambda r: r.score, reverse=True)
