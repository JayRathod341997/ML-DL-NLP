from __future__ import annotations

from dataclasses import dataclass

from tqdm import tqdm

from .config import SearchConfig
from .embedder import SentenceEmbedder
from .vector_store import VectorStore, get_vector_store


@dataclass
class Document:
    doc_id: str
    text: str
    metadata: dict


class DocumentIndexer:
    """Reads documents, embeds them in batches, and persists to a vector store."""

    def __init__(self, config: SearchConfig | None = None) -> None:
        self.config = config or SearchConfig()
        self.embedder = SentenceEmbedder(self.config)
        self.store: VectorStore = get_vector_store(self.config)

    def index(self, documents: list[Document], show_progress: bool = True) -> None:
        """Embed and index a list of Documents.

        Args:
            documents: List of Document objects.
            show_progress: Show progress bar during embedding.
        """
        texts = [doc.text for doc in documents]
        ids = [doc.doc_id for doc in documents]
        metadata = [doc.metadata for doc in documents]

        vectors = self.embedder.encode(texts, show_progress=show_progress)
        self.store.add(ids=ids, vectors=vectors, texts=texts, metadata=metadata)

    def index_batched(self, documents: list[Document], batch_size: int = 512) -> None:
        """Index documents in batches to control memory usage."""
        for i in tqdm(range(0, len(documents), batch_size), desc="Indexing batches"):
            batch = documents[i : i + batch_size]
            self.index(batch, show_progress=False)

    def save(self) -> None:
        """Persist the index to disk."""
        self.store.save()

    def load(self) -> None:
        """Load an existing index from disk."""
        self.store.load()

    def __len__(self) -> int:
        return len(self.store)
