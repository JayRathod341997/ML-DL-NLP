import json
import pytest
from pathlib import Path
from src.data_reader import ChunkedReader


@pytest.fixture
def reader():
    return ChunkedReader(chunk_size=3)


def test_read_jsonl(reader, tmp_path):
    jsonl = tmp_path / "test.jsonl"
    rows = [{"text": f"text {i}"} for i in range(7)]
    jsonl.write_text("\n".join(json.dumps(r) for r in rows))

    chunks = list(reader.read_file(jsonl))
    total = sum(len(c) for c in chunks)
    assert total == 7


def test_read_csv(reader, tmp_path):
    import pandas as pd
    csv_path = tmp_path / "test.csv"
    pd.DataFrame({"text": [f"sample {i}" for i in range(5)]}).to_csv(csv_path, index=False)

    chunks = list(reader.read_file(csv_path))
    total = sum(len(c) for c in chunks)
    assert total == 5


def test_chunks_have_text_key(reader, tmp_path):
    jsonl = tmp_path / "test.jsonl"
    jsonl.write_text("\n".join(json.dumps({"text": f"t{i}"}) for i in range(4)))
    for chunk in reader.read_file(jsonl):
        for row in chunk:
            assert "text" in row


def test_unsupported_format_raises(reader, tmp_path):
    bad_file = tmp_path / "file.xml"
    bad_file.write_text("<root/>")
    with pytest.raises(ValueError):
        list(reader.read_file(bad_file))
