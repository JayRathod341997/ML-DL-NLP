import pytest
from sentence_transformers import InputExample
from unittest.mock import patch, MagicMock

from src.data_loader import load_stsb_pairs


def test_load_stsb_returns_input_examples():
    mock_data = [
        {"sentence1": "A dog runs.", "sentence2": "A dog is running.", "score": 4.0},
        {"sentence1": "The cat sat.", "sentence2": "A bird flew.", "score": 1.0},
    ]

    with patch("src.data_loader.load_dataset", return_value=mock_data):
        examples = load_stsb_pairs(split="test")

    assert len(examples) == 2
    assert all(isinstance(e, InputExample) for e in examples)


def test_stsb_scores_normalised():
    mock_data = [
        {"sentence1": "A", "sentence2": "B", "score": 5.0},
        {"sentence1": "C", "sentence2": "D", "score": 0.0},
    ]
    with patch("src.data_loader.load_dataset", return_value=mock_data):
        examples = load_stsb_pairs(split="test")

    assert abs(examples[0].label - 1.0) < 1e-6
    assert abs(examples[1].label - 0.0) < 1e-6


def test_stsb_scores_in_range():
    mock_data = [{"sentence1": "X", "sentence2": "Y", "score": float(i)} for i in range(6)]
    with patch("src.data_loader.load_dataset", return_value=mock_data):
        examples = load_stsb_pairs(split="test")

    for e in examples:
        assert 0.0 <= e.label <= 1.0
