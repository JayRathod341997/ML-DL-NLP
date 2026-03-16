from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Document:
    text: str
    metadata: dict  # source, page, file_type, etc.


class DocumentLoader:
    """Unified document loader for PDF, HTML, and plain text files."""

    SUPPORTED_EXTENSIONS = {".pdf", ".html", ".htm", ".txt", ".md"}

    def load(self, path: str | Path) -> list[Document]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        ext = path.suffix.lower()
        if ext == ".pdf":
            return self._load_pdf(path)
        elif ext in (".html", ".htm"):
            return self._load_html(path)
        elif ext in (".txt", ".md"):
            return self._load_text(path)
        else:
            raise ValueError(f"Unsupported file type: {ext}. Supported: {self.SUPPORTED_EXTENSIONS}")

    def load_directory(self, dir_path: str | Path) -> list[Document]:
        """Load all supported documents from a directory."""
        dir_path = Path(dir_path)
        docs = []
        for ext in self.SUPPORTED_EXTENSIONS:
            for file in dir_path.rglob(f"*{ext}"):
                docs.extend(self.load(file))
        return docs

    def _load_pdf(self, path: Path) -> list[Document]:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        docs = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                docs.append(
                    Document(
                        text=text,
                        metadata={"source": path.name, "page": i + 1, "file_type": "pdf"},
                    )
                )
        return docs

    def _load_html(self, path: Path) -> list[Document]:
        from bs4 import BeautifulSoup

        html = path.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(html, "lxml")
        # Remove script and style tags
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        return [Document(text=text, metadata={"source": path.name, "file_type": "html"})]

    def _load_text(self, path: Path) -> list[Document]:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return [Document(text=text, metadata={"source": path.name, "file_type": "txt"})]
