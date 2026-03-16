"""Integration test for batch pipeline using mocked model runner."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.config import BatchConfig
from src.pipeline import BatchPipeline


@pytest.fixture
def pipeline_with_mock(tmp_path):
    config = BatchConfig(batch_size=4)
    p = BatchPipeline(config)
    # Mock model runner to avoid downloading models
    mock_runner = MagicMock()
    mock_runner.predict_batch.return_value = [
        {"label": "POSITIVE", "score": 0.9} for _ in range(100)
    ]
    p.model_runner = mock_runner
    return p, tmp_path


def test_run_on_file(pipeline_with_mock, tmp_path):
    pipeline, tmp_path = pipeline_with_mock
    jsonl = tmp_path / "input.jsonl"
    rows = [{"text": f"review {i}"} for i in range(10)]
    jsonl.write_text("\n".join(json.dumps(r) for r in rows))

    output = tmp_path / "output.parquet"
    stats = pipeline.run_on_file(jsonl, output)

    assert stats["total_processed"] == 10
    assert stats["total_errors"] == 0
    assert output.exists()


def test_empty_texts_counted_as_errors(pipeline_with_mock, tmp_path):
    pipeline, tmp_path = pipeline_with_mock
    jsonl = tmp_path / "input.jsonl"
    rows = [{"text": ""}, {"text": "   "}, {"text": "valid text"}]
    jsonl.write_text("\n".join(json.dumps(r) for r in rows))

    output = tmp_path / "output.parquet"
    stats = pipeline.run_on_file(jsonl, output)

    assert stats["total_errors"] == 2
    assert stats["total_processed"] == 1
