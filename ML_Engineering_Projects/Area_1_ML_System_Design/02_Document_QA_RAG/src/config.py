from dataclasses import dataclass, field
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


@dataclass
class RAGConfig:
    embed_model: str = field(
        default_factory=lambda: os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    chroma_persist_dir: Path = field(
        default_factory=lambda: Path(os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db"))
    )
    chroma_collection: str = field(
        default_factory=lambda: os.getenv("CHROMA_COLLECTION", "documents")
    )
    llm_backend: str = field(
        default_factory=lambda: os.getenv("LLM_BACKEND", "ollama")
    )
    ollama_model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    )
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    hf_model: str = field(
        default_factory=lambda: os.getenv("HF_MODEL", "google/flan-t5-base")
    )
    chunk_size: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_SIZE", "512"))
    )
    chunk_overlap: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "50"))
    )
    top_k: int = field(
        default_factory=lambda: int(os.getenv("TOP_K", "5"))
    )
    mmr_fetch_k: int = 20  # fetch more then re-rank with MMR
