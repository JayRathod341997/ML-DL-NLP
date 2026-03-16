from __future__ import annotations

from pathlib import Path
from sentence_transformers import SentenceTransformer


def save_model(model: SentenceTransformer, output_dir: str | Path) -> None:
    """Save a SentenceTransformer model to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(output_dir))
    print(f"Model saved to {output_dir}")


def load_model(model_dir: str | Path) -> SentenceTransformer:
    """Load a SentenceTransformer model from disk."""
    return SentenceTransformer(str(model_dir))


def push_to_hub(model: SentenceTransformer, repo_id: str, token: str | None = None) -> None:
    """Push model to HuggingFace Hub.

    Args:
        model: Trained SentenceTransformer.
        repo_id: HuggingFace Hub repo (e.g., 'username/my-embedding-model').
        token: HuggingFace API token (or set HF_TOKEN env var).
    """
    model.push_to_hub(repo_id, token=token)
    print(f"Model pushed to https://huggingface.co/{repo_id}")


def compare_models(
    model_a: SentenceTransformer,
    model_b: SentenceTransformer,
    queries: list[str],
    documents: list[str],
) -> None:
    """Quick side-by-side comparison of two models on retrieval examples."""
    import numpy as np

    print("Model A vs Model B — Top-3 retrieval comparison\n")
    q_emb_a = model_a.encode(queries, normalize_embeddings=True)
    d_emb_a = model_a.encode(documents, normalize_embeddings=True)
    q_emb_b = model_b.encode(queries, normalize_embeddings=True)
    d_emb_b = model_b.encode(documents, normalize_embeddings=True)

    for i, query in enumerate(queries):
        scores_a = q_emb_a[i] @ d_emb_a.T
        scores_b = q_emb_b[i] @ d_emb_b.T
        top3_a = np.argsort(scores_a)[::-1][:3]
        top3_b = np.argsort(scores_b)[::-1][:3]
        print(f"Query: {query}")
        print(f"  Model A: {[documents[j][:60] for j in top3_a]}")
        print(f"  Model B: {[documents[j][:60] for j in top3_b]}")
        print()
