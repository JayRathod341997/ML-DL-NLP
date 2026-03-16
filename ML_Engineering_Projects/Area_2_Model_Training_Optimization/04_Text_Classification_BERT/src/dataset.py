from __future__ import annotations

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from .config import TrainConfig


class TextClassificationDataset(Dataset):
    """PyTorch Dataset for tokenised text classification.

    Tokenises texts on-the-fly and returns {input_ids, attention_mask, label}.
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: AutoTokenizer,
        max_length: int = 128,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def load_datasets(config: TrainConfig):
    """Load HuggingFace dataset and return train/val/test splits.

    Returns:
        (train_dataset, val_dataset, test_dataset, label_names)
    """
    from datasets import load_dataset
    import numpy as np

    ds = load_dataset(config.dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # Determine text and label column names
    train_split = ds[config.train_split]
    feature_names = list(train_split.features.keys())
    text_col = "text" if "text" in feature_names else feature_names[0]
    label_col = "label" if "label" in feature_names else feature_names[-1]

    # Get label names
    label_feature = train_split.features[label_col]
    label_names = label_feature.names if hasattr(label_feature, "names") else None

    # Train/val split
    train_data = train_split
    val_size = int(len(train_data) * config.val_size)
    indices = np.random.permutation(len(train_data))
    val_indices = indices[:val_size].tolist()
    train_indices = indices[val_size:].tolist()

    train_ds = TextClassificationDataset(
        [train_data[i][text_col] for i in train_indices],
        [train_data[i][label_col] for i in train_indices],
        tokenizer,
        config.max_length,
    )
    val_ds = TextClassificationDataset(
        [train_data[i][text_col] for i in val_indices],
        [train_data[i][label_col] for i in val_indices],
        tokenizer,
        config.max_length,
    )
    test_data = ds[config.test_split]
    test_ds = TextClassificationDataset(
        [row[text_col] for row in test_data],
        [row[label_col] for row in test_data],
        tokenizer,
        config.max_length,
    )
    return train_ds, val_ds, test_ds, label_names
