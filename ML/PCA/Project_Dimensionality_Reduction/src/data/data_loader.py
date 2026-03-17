"""Data loading module for PCA Dimensionality Reduction"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataLoader:
    """Class for loading and preprocessing data for PCA analysis"""

    def __init__(self, config=None):
        self.config = config or {}
        self.scaler = StandardScaler()
        self.data = None
        self.target = None
        self.feature_names = None
        self.target_names = None

    def load_builtin_dataset(self, dataset_name="iris"):
        """Load a built-in dataset for PCA demonstration

        Args:
            dataset_name: Name of the dataset ('iris' or 'breast_cancer')

        Returns:
            X: Feature matrix
            y: Target vector
        """
        if dataset_name == "iris":
            data = load_iris()
            self.feature_names = data.feature_names
            self.target_names = data.target_names
        elif dataset_name == "breast_cancer":
            data = load_breast_cancer()
            self.feature_names = data.feature_names
            self.target_names = data.target_names
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        self.data = data.data
        self.target = data.target

        return self.data, self.target

    def load_csv(self, file_path, target_column=None):
        """Load data from a CSV file

        Args:
            file_path: Path to the CSV file
            target_column: Name of the target column (if any)

        Returns:
            X: Feature matrix
            y: Target vector (if target_column specified)
        """
        df = pd.read_csv(file_path)

        if target_column and target_column in df.columns:
            self.target = df[target_column].values
            X = df.drop(columns=[target_column])
        else:
            self.target = None
            X = df

        self.feature_names = X.columns.tolist()
        self.data = X.values

        return self.data, self.target

    def preprocess(self, apply_scaling=True):
        """Preprocess the data by scaling

        Args:
            apply_scaling: Whether to apply StandardScaler

        Returns:
            X: Preprocessed feature matrix
        """
        if self.data is None:
            raise ValueError(
                "No data loaded. Call load_builtin_dataset or load_csv first."
            )

        if apply_scaling:
            return self.scaler.fit_transform(self.data)
        return self.data

    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into training and testing sets

        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion of data for testing
            random_state: Random seed

        Returns:
            X_train, X_test, y_train, y_test
        """
        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if y is not None else None,
        )

    def get_data_info(self):
        """Get information about the loaded data

        Returns:
            Dictionary with data information
        """
        return {
            "data_shape": self.data.shape if self.data is not None else None,
            "target_shape": self.target.shape if self.target is not None else None,
            "feature_names": self.feature_names,
            "target_names": self.target_names,
            "n_features": len(self.feature_names) if self.feature_names else 0,
        }
