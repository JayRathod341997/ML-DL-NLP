from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from .config import RAGConfig
from .chunker import Chunk


@dataclass
class RetrievedChunk:
    text: str
    metadata: dict
    score: float


class ChromaRetriever:
    """ChromaDB-backed retriever with standard similarity and MMR support."""

    def __init__(self, config: RAGConfig | None = None) -> None:
        import chromadb
        from sentence_transformers import SentenceTransformer

        self.config = config or RAGConfig()
        self._client = chromadb.PersistentClient(path=str(self.config.chroma_persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=self.config.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )
        self._embed_model = SentenceTransformer(self.config.embed_model)

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Add chunks to the vector store."""
        if not chunks:
            return
        texts = [c.text for c in chunks]
        vectors = self._embed_model.encode(texts, normalize_embeddings=True).tolist()
        ids = [f"chunk_{i}_{hash(c.text) % 10**8}" for i, c in enumerate(chunks)]
        metadatas = [c.metadata for c in chunks]
        # Add in batches of 500
        batch = 500
        for i in range(0, len(chunks), batch):
            self._collection.add(
                ids=ids[i : i + batch],
                embeddings=vectors[i : i + batch],
                documents=texts[i : i + batch],
                metadatas=metadatas[i : i + batch],
            )

    def retrieve(self, query: str, k: int | None = None, use_mmr: bool = True) -> list[RetrievedChunk]:
        """Retrieve top-k chunks for a query.

        Args:
            query: Natural language question.
            k: Number of results.
            use_mmr: Use Maximal Marginal Relevance for diversity.
        """
        k = k or self.config.top_k
        n_docs = self._collection.count()
        if n_docs == 0:
            return []

        if use_mmr:
            return self._mmr_retrieve(query, k)
        return self._similarity_retrieve(query, k)

    def _similarity_retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        q_vec = self._embed_model.encode([query], normalize_embeddings=True).tolist()
        results = self._collection.query(
            query_embeddings=q_vec,
            n_results=min(k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        output = []
        for text, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            output.append(RetrievedChunk(text=text, metadata=meta, score=1 - dist))
        return output

    def _mmr_retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        """MMR: fetch fetch_k candidates, then select k diverse results."""
        fetch_k = min(self.config.mmr_fetch_k, self._collection.count())
        q_vec = self._embed_model.encode([query], normalize_embeddings=True)
        results = self._collection.query(
            query_embeddings=q_vec.tolist(),
            n_results=fetch_k,
            include=["documents", "metadatas", "embeddings", "distances"],
        )
        candidates_text = results["documents"][0]
        candidates_meta = results["metadatas"][0]
        candidates_emb = np.array(results["embeddings"][0])
        candidates_score = [1 - d for d in results["distances"][0]]

        selected_indices = self._mmr_select(q_vec[0], candidates_emb, candidates_score, k)
        return [
            RetrievedChunk(
                text=candidates_text[i],
                metadata=candidates_meta[i],
                score=candidates_score[i],
            )
            for i in selected_indices
        ]

    @staticmethod
    def _mmr_select(
        query_vec: np.ndarray,
        candidate_embs: np.ndarray,
        relevance_scores: list[float],
        k: int,
        lambda_mult: float = 0.5,
    ) -> list[int]:
        """Greedy MMR selection: balances relevance and diversity."""
        selected = []
        remaining = list(range(len(candidate_embs)))
        while len(selected) < k and remaining:
            if not selected:
                best = max(remaining, key=lambda i: relevance_scores[i])
            else:
                selected_embs = candidate_embs[selected]
                best_score = -float("inf")
                best = remaining[0]
                for i in remaining:
                    rel = relevance_scores[i]
                    sim_to_selected = float(np.max(candidate_embs[i] @ selected_embs.T))
                    score = lambda_mult * rel - (1 - lambda_mult) * sim_to_selected
                    if score > best_score:
                        best_score = score
                        best = i
            selected.append(best)
            remaining.remove(best)
        return selected

    def count(self) -> int:
        return self._collection.count()
