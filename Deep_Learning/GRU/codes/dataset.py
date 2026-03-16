"""
GRU Dataset Definition
"""

import torch
from torch.utils.data import Dataset
import numpy as np


class SequenceDataset(Dataset):
    def __init__(self, seq_length=20, num_samples=1000):
        self.seq_length = seq_length
        x = np.linspace(0, num_samples * 0.1, num_samples + seq_length)
        self.data = np.sin(x)

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_length]
        y = self.data[idx + self.seq_length]
        return torch.FloatTensor(x).unsqueeze(-1), torch.FloatTensor([y])
