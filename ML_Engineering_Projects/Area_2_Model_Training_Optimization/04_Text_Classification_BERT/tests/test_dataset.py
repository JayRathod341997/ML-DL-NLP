import torch
import pytest
from transformers import AutoTokenizer
from src.dataset import TextClassificationDataset


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("distilbert-base-uncased")


def test_dataset_length(tokenizer):
    ds = TextClassificationDataset(["hello world", "test text"], [0, 1], tokenizer, max_length=32)
    assert len(ds) == 2


def test_dataset_returns_tensors(tokenizer):
    ds = TextClassificationDataset(["hello world"], [2], tokenizer, max_length=32)
    item = ds[0]
    assert "input_ids" in item
    assert "attention_mask" in item
    assert "label" in item


def test_dataset_correct_shapes(tokenizer):
    ds = TextClassificationDataset(["test"], [0], tokenizer, max_length=64)
    item = ds[0]
    assert item["input_ids"].shape == torch.Size([64])
    assert item["attention_mask"].shape == torch.Size([64])
    assert item["label"].shape == torch.Size([])


def test_dataset_label_value(tokenizer):
    ds = TextClassificationDataset(["text"], [3], tokenizer, max_length=32)
    assert ds[0]["label"].item() == 3
