import pytest
from pathlib import Path
from src.document_parser import DocumentParser


@pytest.fixture
def parser():
    return DocumentParser()


def test_parse_txt_file(parser, tmp_path):
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("Hello world. This is a test document.")
    pages = parser.parse(txt_file)
    assert len(pages) == 1
    assert "Hello world" in pages[0].text
    assert pages[0].file_type == "txt"
    assert pages[0].source == "test.txt"


def test_parse_nonexistent_file_raises(parser):
    with pytest.raises(FileNotFoundError):
        parser.parse("/nonexistent/file.txt")


def test_parse_unsupported_extension_raises(parser, tmp_path):
    bad_file = tmp_path / "file.xyz"
    bad_file.write_text("content")
    with pytest.raises(ValueError):
        parser.parse(bad_file)


def test_parse_directory(parser, tmp_path):
    (tmp_path / "a.txt").write_text("Document A content")
    (tmp_path / "b.txt").write_text("Document B content")
    pages = parser.parse_directory(tmp_path)
    assert len(pages) == 2
    sources = {p.source for p in pages}
    assert "a.txt" in sources
    assert "b.txt" in sources
