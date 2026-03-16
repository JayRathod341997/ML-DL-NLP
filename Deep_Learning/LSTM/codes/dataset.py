"""
LSTM Dataset Definition
"""

import torch
from torch.utils.data import Dataset
import numpy as np


class SequenceDataset(Dataset):
    def __init__(self, seq_length=20, num_samples=1000):
        self.seq_length = seq_length
        self.data = self._generate_data(num_samples + seq_length)

    def _generate_data(self, n):
        x = np.linspace(0, n * 0.1, n)
        return np.sin(x) + 0.1 * np.random.randn(n)

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_length]
        y = self.data[idx + self.seq_length]
        x = torch.FloatTensor(x).unsqueeze(-1)
        y = torch.FloatTensor([y])
        return x, y
