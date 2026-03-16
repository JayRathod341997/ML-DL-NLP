import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from sentence_transformers import InputExample, SentenceTransformer

from src.evaluator import STSEvaluator


def make_mock_evaluator():
    pairs = [
        InputExample(texts=["A dog", "A puppy"], label=0.8),
        InputExample(texts=["Rain falls", "The sky is blue"], label=0.1),
        InputExample(texts=["He runs fast", "He sprints quickly"], label=0.9),
    ]
    evaluator = object.__new__(STSEvaluator)
    evaluator.split = "test"
    evaluator._pairs = pairs
    return evaluator


def test_evaluate_returns_expected_keys():
    evaluator = make_mock_evaluator()
    mock_model = MagicMock(spec=SentenceTransformer)
    # Return normalised random vectors
    mock_model.encode = lambda texts, **kwargs: np.random.rand(len(texts), 384).astype(np.float32)

    results = evaluator.evaluate(mock_model)
    assert "spearman_r" in results
    assert "pearson_r" in results
    assert "num_pairs" in results


def test_evaluate_num_pairs():
    evaluator = make_mock_evaluator()
    mock_model = MagicMock(spec=SentenceTransformer)
    mock_model.encode = lambda texts, **kwargs: np.random.rand(len(texts), 384).astype(np.float32)

    results = evaluator.evaluate(mock_model)
    assert results["num_pairs"] == 3
