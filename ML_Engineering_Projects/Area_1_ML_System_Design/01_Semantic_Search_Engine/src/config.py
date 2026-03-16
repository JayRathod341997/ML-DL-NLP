from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv()


@dataclass
class SearchConfig:
    embed_model: str = field(
        default_factory=lambda: os.getenv(
            "EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    index_path: Path = field(
        default_factory=lambda: Path(os.getenv("INDEX_PATH", "./data/faiss_index"))
    )
    vector_store: str = field(
        default_factory=lambda: os.getenv("VECTOR_STORE", "faiss")
    )
    chroma_persist_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
        )
    )
    chroma_collection: str = field(
        default_factory=lambda: os.getenv("CHROMA_COLLECTION", "documents")
    )
    top_k: int = field(
        default_factory=lambda: int(os.getenv("TOP_K", "10"))
    )
    batch_size: int = 64
    normalize_embeddings: bool = True
    device: str = "cpu"
