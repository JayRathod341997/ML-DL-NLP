from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ParsedPage:
    text: str
    page_number: int
    source: str
    file_type: str


class DocumentParser:
    """Unified parser for PDF, DOCX, and plain text documents."""

    SUPPORTED = {".pdf", ".docx", ".txt", ".md"}

    def parse(self, path: str | Path) -> list[ParsedPage]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"{path}")
        ext = path.suffix.lower()
        if ext == ".pdf":
            return self._parse_pdf(path)
        elif ext == ".docx":
            return self._parse_docx(path)
        elif ext in (".txt", ".md"):
            return self._parse_text(path)
        else:
            raise ValueError(f"Unsupported: {ext}")

    def parse_directory(self, dir_path: str | Path) -> list[ParsedPage]:
        dir_path = Path(dir_path)
        pages = []
        for ext in self.SUPPORTED:
            for f in dir_path.rglob(f"*{ext}"):
                pages.extend(self.parse(f))
        return pages

    def _parse_pdf(self, path: Path) -> list[ParsedPage]:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return [
            ParsedPage(
                text=page.extract_text() or "",
                page_number=i + 1,
                source=path.name,
                file_type="pdf",
            )
            for i, page in enumerate(reader.pages)
            if (page.extract_text() or "").strip()
        ]

    def _parse_docx(self, path: Path) -> list[ParsedPage]:
        from docx import Document
        doc = Document(str(path))
        full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return [ParsedPage(text=full_text, page_number=1, source=path.name, file_type="docx")]

    def _parse_text(self, path: Path) -> list[ParsedPage]:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return [ParsedPage(text=text, page_number=1, source=path.name, file_type="txt")]
