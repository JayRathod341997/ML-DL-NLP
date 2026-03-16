from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import SearchConfig


@dataclass
class SearchResult:
    doc_id: str
    text: str
    metadata: dict
    score: float


class VectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    def add(self, ids: list[str], vectors: np.ndarray, texts: list[str], metadata: list[dict]) -> None:
        ...

    @abstractmethod
    def search(self, query_vector: np.ndarray, k: int) -> list[SearchResult]:
        ...

    @abstractmethod
    def save(self) -> None:
        ...

    @abstractmethod
    def load(self) -> None:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...


class FAISSStore(VectorStore):
    """FAISS-backed vector store using inner-product (cosine on normalised vecs)."""

    def __init__(self, config: SearchConfig) -> None:
        import faiss

        self.config = config
        self.dim: int | None = None
        self.index = None
        self._id_map: list[str] = []
        self._texts: list[str] = []
        self._metadata: list[dict] = []
        self._meta_path = Path(config.index_path).with_suffix(".meta.json")

    def _init_index(self, dim: int) -> None:
        import faiss

        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)

    def add(self, ids: list[str], vectors: np.ndarray, texts: list[str], metadata: list[dict]) -> None:
        if self.index is None:
            self._init_index(vectors.shape[1])
        self.index.add(vectors)
        self._id_map.extend(ids)
        self._texts.extend(texts)
        self._metadata.extend(metadata)

    def search(self, query_vector: np.ndarray, k: int) -> list[SearchResult]:
        if self.index is None or len(self) == 0:
            return []
        k = min(k, len(self))
        scores, indices = self.index.search(query_vector, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append(
                SearchResult(
                    doc_id=self._id_map[idx],
                    text=self._texts[idx],
                    metadata=self._metadata[idx],
                    score=float(score),
                )
            )
        return results

    def save(self) -> None:
        import faiss

        path = Path(self.config.index_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))
        with open(self._meta_path, "w") as f:
            json.dump({"ids": self._id_map, "texts": self._texts, "metadata": self._metadata}, f)

    def load(self) -> None:
        import faiss

        path = Path(self.config.index_path)
        self.index = faiss.read_index(str(path))
        with open(self._meta_path) as f:
            data = json.load(f)
        self._id_map = data["ids"]
        self._texts = data["texts"]
        self._metadata = data["metadata"]
        self.dim = self.index.d

    def __len__(self) -> int:
        return len(self._id_map)


class ChromaStore(VectorStore):
    """ChromaDB-backed vector store."""

    def __init__(self, config: SearchConfig) -> None:
        import chromadb

        self.config = config
        self._client = chromadb.PersistentClient(path=str(config.chroma_persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=config.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, ids: list[str], vectors: np.ndarray, texts: list[str], metadata: list[dict]) -> None:
        self._collection.add(
            ids=ids,
            embeddings=vectors.tolist(),
            documents=texts,
            metadatas=metadata,
        )

    def search(self, query_vector: np.ndarray, k: int) -> list[SearchResult]:
        results = self._collection.query(
            query_embeddings=query_vector.tolist(),
            n_results=min(k, len(self)),
            include=["documents", "metadatas", "distances"],
        )
        output = []
        for doc_id, text, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append(
                SearchResult(doc_id=doc_id, text=text, metadata=meta, score=1 - dist)
            )
        return output

    def save(self) -> None:
        pass  # ChromaDB persists automatically

    def load(self) -> None:
        pass  # ChromaDB loads automatically

    def __len__(self) -> int:
        return self._collection.count()


def get_vector_store(config: SearchConfig) -> VectorStore:
    """Factory function to get the configured vector store."""
    if config.vector_store == "faiss":
        return FAISSStore(config)
    elif config.vector_store == "chroma":
        return ChromaStore(config)
    else:
        raise ValueError(f"Unknown vector store: {config.vector_store}. Use 'faiss' or 'chroma'.")
