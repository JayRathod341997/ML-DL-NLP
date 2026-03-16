from __future__ import annotations

import re


class TextPreprocessor:
    """Light text cleaning before tokenisation."""

    MAX_LENGTH_CHARS = 10_000  # truncate very long texts before tokenisation

    def process(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""
        text = text[: self.MAX_LENGTH_CHARS]
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def process_batch(self, texts: list[str]) -> list[str]:
        return [self.process(t) for t in texts]
