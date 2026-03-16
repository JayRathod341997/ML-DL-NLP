"""
Dataset Module for CNN

Data loading utilities for CNN training with image transforms.
"""

from pathlib import Path
from typing import Callable

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class ImageDataset(Dataset):
    """
    Custom dataset for loading images from a directory structure.

    Expected structure:
        root/
            class1/
                img1.jpg
                img2.jpg
            class2/
                img1.jpg
    """

    def __init__(
        self,
        root: str | Path,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ) -> None:
        """
        Initialize the dataset.

        Args:
            root: Root directory containing class subdirectories
            transform: Transform to apply to images
            target_transform: Transform to apply to labels
        """
        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform

        # Find all class directories
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Collect all image paths and labels
        self.samples = []
        for class_name in self.classes:
            class_dir = self.root / class_name
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (image, label)
        """
        from PIL import Image

        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label


class CIFAR10DataModule:
    """
    Data module for CIFAR-10 dataset.
    """

    def __init__(
        self,
        data_dir: str | Path = "./datasets",
        batch_size: int = 64,
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

        # Data transforms
        self.train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
                ),
            ]
        )

        self.test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
                ),
            ]
        )

    def get_train_loader(self) -> DataLoader:
        """Get training DataLoader."""
        train_dataset = datasets.CIFAR10(
            self.data_dir,
            train=True,
            download=True,
            transform=self.train_transform,
        )

        # Split for validation
        n_val = int(len(train_dataset) * self.val_split)
        n_train = len(train_dataset) - n_val

        train_subset, _ = torch.utils.data.random_split(train_dataset, [n_train, n_val])

        return DataLoader(
            train_subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )

    def get_val_loader(self) -> DataLoader:
        """Get validation DataLoader."""
        train_dataset = datasets.CIFAR10(
            self.data_dir,
            train=True,
            download=True,
            transform=self.test_transform,
        )

        n_val = int(len(train_dataset) * self.val_split)
        n_train = len(train_dataset) - n_val

        _, val_subset = torch.utils.data.random_split(train_dataset, [n_train, n_val])

        return DataLoader(
            val_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

    def get_test_loader(self) -> DataLoader:
        """Get test DataLoader."""
        test_dataset = datasets.CIFAR10(
            self.data_dir,
            train=False,
            download=True,
            transform=self.test_transform,
        )

        return DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )


class MNISTDataModule:
    """
    Data module for MNIST dataset (for CNN training).
    """

    def __init__(
        self,
        data_dir: str | Path = "./datasets",
        batch_size: int = 64,
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

        # Data transforms with augmentation for training
        self.train_transform = transforms.Compose(
            [
                transforms.RandomRotation(10),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        self.test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

    def get_train_loader(self) -> DataLoader:
        """Get training DataLoader."""
        train_dataset = datasets.MNIST(
            self.data_dir,
            train=True,
            download=True,
            transform=self.train_transform,
        )

        n_val = int(len(train_dataset) * self.val_split)
        n_train = len(train_dataset) - n_val

        train_subset, _ = torch.utils.data.random_split(train_dataset, [n_train, n_val])

        return DataLoader(
            train_subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )

    def get_val_loader(self) -> DataLoader:
        """Get validation DataLoader."""
        train_dataset = datasets.MNIST(
            self.data_dir,
            train=True,
            download=True,
            transform=self.test_transform,
        )

        n_val = int(len(train_dataset) * self.val_split)
        n_train = len(train_dataset) - n_val

        _, val_subset = torch.utils.data.random_split(train_dataset, [n_train, n_val])

        return DataLoader(
            val_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

    def get_test_loader(self) -> DataLoader:
        """Get test DataLoader."""
        test_dataset = datasets.MNIST(
            self.data_dir,
            train=False,
            download=True,
            transform=self.test_transform,
        )

        return DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )


def get_standard_transforms(
    image_size: int = 224,
    augment: bool = True,
) -> tuple[transforms.Compose, transforms.Compose]:
    """
    Get standard transforms for image classification.

    Args:
        image_size: Target image size
        augment: Whether to include data augmentation

    Returns:
        Tuple of (train_transform, val_transform)
    """
    # ImageNet normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if augment:
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize,
            ]
        )

    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )

    return train_transform, val_transform
