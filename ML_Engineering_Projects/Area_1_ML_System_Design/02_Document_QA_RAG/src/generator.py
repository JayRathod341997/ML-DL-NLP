from __future__ import annotations

from .config import RAGConfig

PROMPT_TEMPLATE = """You are a helpful assistant. Answer the question based ONLY on the provided context.
If the answer is not in the context, say "I don't have enough information to answer this."

Context:
{context}

Question: {question}

Answer:"""


class LLMGenerator:
    """Generates answers using either Ollama (local) or a HuggingFace model."""

    def __init__(self, config: RAGConfig | None = None) -> None:
        self.config = config or RAGConfig()

    def generate(self, question: str, context_chunks: list[str]) -> str:
        """Generate an answer given a question and list of context chunks.

        Args:
            question: The user's question.
            context_chunks: Retrieved text chunks to ground the answer.

        Returns:
            Generated answer string.
        """
        context = "\n\n---\n\n".join(context_chunks)
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)

        if self.config.llm_backend == "ollama":
            return self._generate_ollama(prompt)
        elif self.config.llm_backend == "huggingface":
            return self._generate_hf(prompt)
        else:
            raise ValueError(f"Unknown LLM backend: {self.config.llm_backend}")

    def _generate_ollama(self, prompt: str) -> str:
        import ollama

        response = ollama.generate(
            model=self.config.ollama_model,
            prompt=prompt,
            options={"temperature": 0.1, "num_predict": 512},
        )
        return response["response"].strip()

    def _generate_hf(self, prompt: str) -> str:
        from transformers import pipeline

        pipe = pipeline(
            "text2text-generation",
            model=self.config.hf_model,
            max_new_tokens=256,
        )
        result = pipe(prompt, truncation=True)
        return result[0]["generated_text"].strip()
