from __future__ import annotations

from dataclasses import dataclass

from .config import RAGConfig
from .retriever import ChromaRetriever, RetrievedChunk
from .generator import LLMGenerator


@dataclass
class RAGResponse:
    question: str
    answer: str
    sources: list[str]
    context_chunks: list[RetrievedChunk]


class RAGPipeline:
    """End-to-end RAG pipeline: retrieval + generation."""

    def __init__(self, config: RAGConfig | None = None) -> None:
        self.config = config or RAGConfig()
        self.retriever = ChromaRetriever(self.config)
        self.generator = LLMGenerator(self.config)

    def ask(self, question: str) -> RAGResponse:
        """Ask a question and get a grounded answer.

        Args:
            question: Natural language question.

        Returns:
            RAGResponse with answer, sources, and context chunks used.
        """
        chunks = self.retriever.retrieve(question, use_mmr=True)
        context_texts = [c.text for c in chunks]
        answer = self.generator.generate(question, context_texts)
        sources = list({c.metadata.get("source", "unknown") for c in chunks})
        return RAGResponse(
            question=question,
            answer=answer,
            sources=sources,
            context_chunks=chunks,
        )
