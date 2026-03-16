import torch
import pytest
from src.config import TrainConfig
from src.model import BERTClassifier


@pytest.fixture(scope="module")
def model():
    config = TrainConfig(model_name="distilbert-base-uncased", num_labels=4)
    return BERTClassifier(config)


def test_model_forward_shape(model):
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, 30522, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    logits = model(input_ids, attention_mask)
    assert logits.shape == torch.Size([batch_size, 4])


def test_model_output_not_nan(model):
    input_ids = torch.randint(0, 30522, (1, 32))
    attention_mask = torch.ones(1, 32, dtype=torch.long)
    logits = model(input_ids, attention_mask)
    assert not torch.isnan(logits).any()


def test_model_different_num_labels():
    config = TrainConfig(model_name="distilbert-base-uncased", num_labels=14)
    m = BERTClassifier(config)
    input_ids = torch.randint(0, 30522, (1, 16))
    attention_mask = torch.ones(1, 16, dtype=torch.long)
    logits = m(input_ids, attention_mask)
    assert logits.shape == torch.Size([1, 14])
