"""Integration test for RAG pipeline (mocks the LLM generator)."""

import pytest
from unittest.mock import patch, MagicMock

from src.config import RAGConfig
from src.chunker import Chunk
from src.retriever import ChromaRetriever
from src.rag_pipeline import RAGPipeline


@pytest.fixture
def pipeline(tmp_path):
    config = RAGConfig(
        chroma_persist_dir=tmp_path / "chroma",
        chroma_collection="test_rag",
    )
    p = RAGPipeline(config)
    # Pre-load some chunks
    chunks = [
        Chunk("Transformers use self-attention mechanisms.", {"source": "paper.pdf", "chunk_index": 0}),
        Chunk("BERT is a pre-trained transformer model.", {"source": "paper.pdf", "chunk_index": 1}),
    ]
    p.retriever.add_chunks(chunks)
    return p


def test_ask_returns_rag_response(pipeline):
    with patch.object(pipeline.generator, "generate", return_value="Mocked answer."):
        response = pipeline.ask("What is BERT?")
    assert response.question == "What is BERT?"
    assert response.answer == "Mocked answer."
    assert len(response.context_chunks) > 0
    assert "paper.pdf" in response.sources


def test_ask_passes_context_to_generator(pipeline):
    captured = {}

    def fake_generate(question, context_chunks):
        captured["context"] = context_chunks
        return "answer"

    with patch.object(pipeline.generator, "generate", side_effect=fake_generate):
        pipeline.ask("What are transformers?")

    assert len(captured["context"]) > 0
