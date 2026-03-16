from __future__ import annotations

from dataclasses import dataclass

from .config import RAGConfig
from .document_loader import Document


@dataclass
class Chunk:
    text: str
    metadata: dict  # inherits parent doc metadata + chunk_index


class RecursiveChunker:
    """Splits documents into overlapping chunks using a hierarchy of separators.

    Splits by: paragraph (\\n\\n), newline (\\n), space ( ), then characters.
    This preserves semantic units (paragraphs) where possible.
    """

    SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, config: RAGConfig | None = None) -> None:
        self.config = config or RAGConfig()

    def chunk(self, document: Document) -> list[Chunk]:
        """Split a single document into chunks."""
        chunks = self._split(document.text, self.SEPARATORS)
        return [
            Chunk(
                text=chunk_text,
                metadata={**document.metadata, "chunk_index": i},
            )
            for i, chunk_text in enumerate(chunks)
        ]

    def chunk_all(self, documents: list[Document]) -> list[Chunk]:
        """Split a list of documents into chunks."""
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.chunk(doc))
        return all_chunks

    def _split(self, text: str, separators: list[str]) -> list[str]:
        if not separators:
            return self._fixed_split(text)
        sep = separators[0]
        parts = text.split(sep) if sep else list(text)
        merged = self._merge_splits(parts, sep)
        result = []
        for part in merged:
            if len(part) > self.config.chunk_size:
                result.extend(self._split(part, separators[1:]))
            else:
                result.append(part)
        return [r for r in result if r.strip()]

    def _merge_splits(self, splits: list[str], separator: str) -> list[str]:
        merged = []
        current = ""
        for split in splits:
            candidate = (current + separator + split).strip() if current else split
            if len(candidate) <= self.config.chunk_size:
                current = candidate
            else:
                if current:
                    merged.append(current)
                # Handle overlap
                if self.config.chunk_overlap > 0 and current:
                    overlap_text = current[-self.config.chunk_overlap:]
                    current = (overlap_text + separator + split).strip()
                else:
                    current = split
        if current:
            merged.append(current)
        return merged

    def _fixed_split(self, text: str) -> list[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.config.chunk_size
            chunks.append(text[start:end])
            start = end - self.config.chunk_overlap
        return chunks
