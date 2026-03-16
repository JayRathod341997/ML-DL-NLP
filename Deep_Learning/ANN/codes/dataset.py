"""
Dataset Module

Data loading utilities for neural network training.
Includes DataLoaders, transforms, and helper functions.
"""

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset


class TabularDataset(Dataset):
    """
    Dataset for tabular data from CSV files.

    Handles loading features and labels from CSV files with support
    for train/test splits and normalization.

    Example:
        >>> dataset = TabularDataset(
        ...     features_path="train.csv",
        ...     label_column="label",
        ...     transform=normalize
        ... )
        >>> loader = DataLoader(dataset, batch_size=32, shuffle=True)
    """

    def __init__(
        self,
        features: np.ndarray | pd.DataFrame,
        labels: np.ndarray | pd.Series | None = None,
        transform: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        """
        Initialize the dataset.

        Args:
            features: Input features as numpy array or DataFrame
            labels: Target labels (optional for test data)
            transform: Optional transform to apply to features
        """
        self.features = torch.FloatTensor(
            features.values if isinstance(features, pd.DataFrame) else features
        )

        self.labels = (
            torch.LongTensor(labels.values if isinstance(labels, pd.Series) else labels)
            if labels is not None
            else None
        )

        self.transform = transform

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor] | tuple[Tensor, ...]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (features, labels) or just features if no labels
        """
        x = self.features[idx]

        if self.transform:
            x = self.transform(x)

        if self.labels is not None:
            return x, self.labels[idx]

        return (x,)


class IrisDataset(TensorDataset):
    """
    Dataset for the Iris flower dataset.

    Provides easy loading and splitting of the classic Iris dataset.

    Attributes:
        FEATURE_NAMES: List of feature names
        CLASS_NAMES: List of class names
    """

    FEATURE_NAMES = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
    ]
    CLASS_NAMES = ["setosa", "versicolor", "virginica"]

    @staticmethod
    def load_from_csv(
        filepath: str | Path, test_split: float = 0.2
    ) -> tuple[DataLoader, DataLoader]:
        """
        Load Iris dataset from CSV and create train/test loaders.

        Args:
            filepath: Path to the CSV file
            test_split: Fraction of data for testing

        Returns:
            Tuple of (train_loader, test_loader)
        """
        df = pd.read_csv(
            filepath, header=None, names=IrisDataset.FEATURE_NAMES + ["target"]
        )

        # Convert to tensors
        X = torch.FloatTensor(df[IrisDataset.FEATURE_NAMES].values)
        y = torch.LongTensor(df["target"].values)

        # Normalize features
        mean = X.mean(dim=0)
        std = X.std(dim=0)
        X = (X - mean) / std

        # Split
        split_idx = int(len(X) * (1 - test_split))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        return train_loader, test_loader


class MNISTDataModule:
    """
    Data module for MNIST dataset.

    Provides train/val/test DataLoaders with appropriate transforms.
    """

    def __init__(
        self,
        data_dir: str | Path = "./datasets",
        batch_size: int = 32,
        val_split: float = 0.1,
    ) -> None:
        """
        Initialize the data module.

        Args:
            data_dir: Directory to store/load data
            batch_size: Batch size for DataLoaders
            val_split: Fraction of training data for validation
        """
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.val_split = val_split

    def get_train_loader(self) -> DataLoader:
        """Get training DataLoader."""
        from torchvision import datasets, transforms

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        train_dataset = datasets.MNIST(
            self.data_dir,
            train=True,
            download=True,
            transform=transform,
        )

        # Split for validation
        n_val = int(len(train_dataset) * self.val_split)
        n_train = len(train_dataset) - n_val

        train_subset, val_subset = torch.utils.data.random_split(
            train_dataset, [n_train, n_val]
        )

        return DataLoader(
            train_subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )

    def get_val_loader(self) -> DataLoader:
        """Get validation DataLoader."""
        from torchvision import datasets, transforms

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        train_dataset = datasets.MNIST(
            self.data_dir,
            train=True,
            download=True,
            transform=transform,
        )

        n_val = int(len(train_dataset) * self.val_split)
        n_train = len(train_dataset) - n_val

        train_subset, val_subset = torch.utils.data.random_split(
            train_dataset, [n_train, n_val]
        )

        return DataLoader(
            val_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

    def get_test_loader(self) -> DataLoader:
        """Get test DataLoader."""
        from torchvision import datasets, transforms

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        test_dataset = datasets.MNIST(
            self.data_dir,
            train=False,
            download=True,
            transform=transform,
        )

        return DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )


def load_csv_dataset(
    filepath: str | Path,
    label_column: str,
    test_split: float = 0.2,
    batch_size: int = 32,
    normalize: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """
    Load a CSV dataset and create train/test DataLoaders.

    Args:
        filepath: Path to CSV file
        label_column: Name of the label column
        test_split: Fraction for test set
        batch_size: Batch size
        normalize: Whether to normalize features

    Returns:
        Tuple of (train_loader, test_loader)
    """
    df = pd.read_csv(filepath)

    # Separate features and labels
    X = df.drop(columns=[label_column])
    y = df[label_column]

    # Convert to tensors
    X_tensor = torch.FloatTensor(X.values)
    y_tensor = torch.LongTensor(y.values)

    # Normalize if requested
    if normalize:
        mean = X_tensor.mean(dim=0)
        std = X_tensor.std(dim=0)
        std[std == 0] = 1  # Avoid division by zero
        X_tensor = (X_tensor - mean) / std

    # Split data
    split_idx = int(len(X_tensor) * (1 - test_split))
    X_train, X_test = X_tensor[:split_idx], X_tensor[split_idx:]
    y_train, y_test = y_tensor[:split_idx], y_tensor[split_idx:]

    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
