"""Data loading module for Random Forest Fraud Detection"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataLoader:
    """Class for loading and preprocessing fraud detection data"""

    def __init__(self, config=None):
        self.config = config or {}
        self.scaler = StandardScaler()
        self.X = None
        self.y = None

    def load_builtin_dataset(self, dataset_name="creditcard"):
        """Load a built-in dataset for fraud detection

        Args:
            dataset_name: Name of the dataset

        Returns:
            X: Feature matrix
            y: Target vector
        """
        # Generate synthetic credit card data
        X, y = make_classification(
            n_samples=10000,
            n_features=30,
            n_informative=25,
            n_redundant=5,
            random_state=42,
            weights=[0.95, 0.05],  # Imbalanced classes
        )

        self.X = X
        self.y = y
        return self.X, self.y

    def load_csv(self, file_path, target_column="Class"):
        """Load data from a CSV file"""
        df = pd.read_csv(file_path)
        if target_column in df.columns:
            self.y = df[target_column].values
            self.X = df.drop(columns=[target_column]).values
        return self.X, self.y

    def preprocess(self, apply_scaling=True):
        """Preprocess the data"""
        if self.X is None:
            raise ValueError("No data loaded")
        if apply_scaling:
            return self.scaler.fit_transform(self.X)
        return self.X

    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        return train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

    def get_data_info(self):
        """Get information about the loaded data"""
        return {
            "X_shape": self.X.shape if self.X is not None else None,
            "y_shape": self.y.shape if self.y is not None else None,
            "class_distribution": (
                dict(zip(*np.unique(self.y, return_counts=True)))
                if self.y is not None
                else None
            ),
        }
