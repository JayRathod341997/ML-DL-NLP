"""
RNN Dataset Definition
"""

import torch
from torch.utils.data import Dataset
import numpy as np


class SequenceDataset(Dataset):
    """Generate sine wave sequences for prediction."""

    def __init__(self, seq_length=20, num_samples=1000):
        self.seq_length = seq_length
        self.num_samples = num_samples

        # Generate sine wave data
        self.data = self._generate_data(num_samples + seq_length)

    def _generate_data(self, n):
        """Generate sine wave data."""
        x = np.linspace(0, n * 0.1, n)
        return np.sin(x)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Input sequence
        x = self.data[idx : idx + self.seq_length]
        # Target: next value
        y = self.data[idx + self.seq_length]

        # Reshape for RNN: (seq_length, input_size)
        x = torch.FloatTensor(x).unsqueeze(-1)  # Add feature dimension
        y = torch.FloatTensor([y])

        return x, y


class TextSequenceDataset(Dataset):
    """Text sequence dataset for NLP tasks."""

    def __init__(self, texts, word2idx, seq_length=50):
        self.texts = texts
        self.word2idx = word2idx
        self.seq_length = seq_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # Convert words to indices
        indices = [self.word2idx.get(word, 0) for word in text.split()]

        # Pad or truncate
        if len(indices) < self.seq_length:
            indices += [0] * (self.seq_length - len(indices))
        else:
            indices = indices[: self.seq_length]

        x = torch.LongTensor(indices[:-1])
        y = torch.LongTensor(indices[1:])

        return x, y


class TimeSeriesDataset(Dataset):
    """Time series dataset for forecasting."""

    def __init__(self, data, seq_length=24, pred_length=1):
        self.data = torch.FloatTensor(data)
        self.seq_length = seq_length
        self.pred_length = pred_length

    def __len__(self):
        return len(self.data) - self.seq_length - self.pred_length + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_length]
        y = self.data[idx + self.seq_length : idx + self.seq_length + self.pred_length]

        # Add feature dimension
        x = x.unsqueeze(-1)
        y = y.squeeze()

        return x, y
