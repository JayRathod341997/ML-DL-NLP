from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .config import SearchConfig


class SentenceEmbedder:
    """Wraps a SentenceTransformer model for batched text encoding."""

    def __init__(self, config: SearchConfig | None = None) -> None:
        self.config = config or SearchConfig()
        self.model = SentenceTransformer(
            self.config.embed_model, device=self.config.device
        )
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def encode(
        self,
        texts: list[str],
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode a list of texts into L2-normalised embedding vectors.

        Args:
            texts: List of strings to encode.
            show_progress: Show tqdm progress bar.

        Returns:
            Float32 numpy array of shape (len(texts), embedding_dim).
        """
        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize_embeddings,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string. Returns shape (1, embedding_dim)."""
        return self.encode([query])
