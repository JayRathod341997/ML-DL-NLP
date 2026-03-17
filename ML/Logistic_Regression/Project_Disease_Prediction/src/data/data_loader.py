"""Data loading module for Logistic Regression Disease Prediction"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataLoader:
    """Class for loading and preprocessing disease prediction data"""

    def __init__(self, config=None):
        self.config = config or {}
        self.scaler = StandardScaler()
        self.X = None
        self.y = None
        self.feature_names = None
        self.target_names = None

    def load_builtin_dataset(self, dataset_name="breast_cancer"):
        """Load a built-in dataset for disease prediction

        Args:
            dataset_name: Name of the dataset ('breast_cancer' or 'diabetes')

        Returns:
            X: Feature matrix
            y: Target vector
        """
        if dataset_name == "breast_cancer":
            data = load_breast_cancer()
            self.feature_names = data.feature_names
            self.target_names = data.target_names
        elif dataset_name == "diabetes":
            data = load_diabetes()
            self.feature_names = [f"feature_{i}" for i in range(data.data.shape[1])]
            self.target_names = ["target"]
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        self.X = data.data
        self.y = data.target

        return self.X, self.y

    def load_csv(self, file_path, target_column="target"):
        """Load data from a CSV file

        Args:
            file_path: Path to the CSV file
            target_column: Name of the target column

        Returns:
            X: Feature matrix
            y: Target vector
        """
        df = pd.read_csv(file_path)

        if target_column and target_column in df.columns:
            self.y = df[target_column].values
            X = df.drop(columns=[target_column])
        else:
            self.y = None
            X = df

        self.feature_names = X.columns.tolist()
        self.X = X.values

        return self.X, self.y

    def preprocess(self, apply_scaling=True):
        """Preprocess the data by scaling

        Args:
            apply_scaling: Whether to apply StandardScaler

        Returns:
            X: Preprocessed feature matrix
        """
        if self.X is None:
            raise ValueError(
                "No data loaded. Call load_builtin_dataset or load_csv first."
            )

        if apply_scaling:
            return self.scaler.fit_transform(self.X)
        return self.X

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
            "X_shape": self.X.shape if self.X is not None else None,
            "y_shape": self.y.shape if self.y is not None else None,
            "feature_names": self.feature_names,
            "target_names": self.target_names,
            "n_features": len(self.feature_names) if self.feature_names else 0,
            "class_distribution": (
                dict(zip(*np.unique(self.y, return_counts=True)))
                if self.y is not None
                else None
            ),
        }
