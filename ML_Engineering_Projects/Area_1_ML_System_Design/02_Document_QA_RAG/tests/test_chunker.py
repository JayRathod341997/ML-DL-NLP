import pytest
from src.config import RAGConfig
from src.chunker import RecursiveChunker
from src.document_loader import Document


@pytest.fixture
def chunker():
    config = RAGConfig(chunk_size=100, chunk_overlap=20)
    return RecursiveChunker(config)


def test_chunk_single_doc(chunker):
    doc = Document(text="Hello world. " * 50, metadata={"source": "test"})
    chunks = chunker.chunk(doc)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= 150  # some tolerance for split boundaries


def test_chunk_preserves_metadata(chunker):
    doc = Document(text="Some text.", metadata={"source": "doc.pdf", "page": 1})
    chunks = chunker.chunk(doc)
    for chunk in chunks:
        assert chunk.metadata["source"] == "doc.pdf"
        assert chunk.metadata["page"] == 1
        assert "chunk_index" in chunk.metadata


def test_chunk_index_increments(chunker):
    doc = Document(text="paragraph one\n\n" * 20, metadata={"source": "test"})
    chunks = chunker.chunk(doc)
    indices = [c.metadata["chunk_index"] for c in chunks]
    assert indices == list(range(len(chunks)))


def test_chunk_all_multiple_docs(chunker):
    docs = [
        Document(text="Doc A content. " * 20, metadata={"source": "a.txt"}),
        Document(text="Doc B content. " * 20, metadata={"source": "b.txt"}),
    ]
    chunks = chunker.chunk_all(docs)
    sources = {c.metadata["source"] for c in chunks}
    assert "a.txt" in sources
    assert "b.txt" in sources
