"""Data loading module for KNN Customer Segmentation"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler


class DataLoader:
    """Class for loading and preprocessing customer data"""

    def __init__(self, config=None):
        self.config = config or {}
        self.scaler = StandardScaler()
        self.X = None
        self.y = None

    def load_builtin_dataset(self, n_samples=500, n_features=5, centers=5):
        """Generate synthetic customer data"""
        X, y = make_blobs(
            n_samples=n_samples, n_features=n_features, centers=centers, random_state=42
        )
        self.X = X
        self.y = y
        return self.X, self.y

    def load_csv(self, file_path):
        """Load data from CSV"""
        df = pd.read_csv(file_path)
        self.X = df.values
        return self.X, None

    def preprocess(self, apply_scaling=True):
        """Preprocess data"""
        if self.X is None:
            raise ValueError("No data loaded")
        if apply_scaling:
            return self.scaler.fit_transform(self.X)
        return self.X

    def get_data_info(self):
        return {"X_shape": self.X.shape if self.X is not None else None}
