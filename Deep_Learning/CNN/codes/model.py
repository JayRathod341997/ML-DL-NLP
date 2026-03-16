"""
CNN Model Implementation

A modular PyTorch implementation of Convolutional Neural Networks
for image classification tasks. Uses device-agnostic setup for CPU/GPU.
"""

from torch import nn, Tensor
from typing import Sequence


class CNN(nn.Module):
    """
    A Convolutional Neural Network for image classification.

    Architecture:
        Conv → ReLU → Pool → Conv → ReLU → Pool → FC → Output

    Attributes:
        features: Convolutional feature extractor
        classifier: Fully connected classifier head

    Example:
        >>> model = CNN(num_classes=10)
        >>> x = torch.randn(1, 1, 28, 28)
        >>> output = model(x)  # Shape: (1, 10)
    """

    def __init__(
        self,
        num_classes: int = 10,
        input_channels: int = 1,
        dropout_rate: float = 0.25,
    ) -> None:
        """
        Initialize the CNN model.

        Args:
            num_classes: Number of output classes
            input_channels: Number of input channels (1 for grayscale, 3 for RGB)
            dropout_rate: Dropout probability for regularization
        """
        super().__init__()

        # Feature extractor
        self.features = nn.Sequential(
            # Conv Block 1: 1x28x28 -> 32x14x14
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate / 2),
            # Conv Block 2: 32x14x14 -> 64x7x7
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate / 2),
            # Conv Block 3: 64x7x7 -> 128x3x3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        x = self.features(x)
        x = self.classifier(x)
        return x

    def predict(self, x: Tensor) -> Tensor:
        """
        Get class predictions.

        Args:
            x: Input tensor

        Returns:
            Predicted class indices
        """
        logits = self.forward(x)
        return logits.argmax(dim=1)

    def predict_proba(self, x: Tensor) -> Tensor:
        """
        Get class probabilities using softmax.

        Args:
            x: Input tensor

        Returns:
            Probability distribution over classes
        """
        logits = self.forward(x)
        return logits.softmax(dim=1)


class LeNet5(nn.Module):
    """
    Implementation of the classic LeNet-5 architecture.

    Architecture:
        Conv → AvgPool → Conv → AvgPool → FC → FC → Output

    Reference: LeCun et al., "Gradient-Based Learning Applied to Document Recognition"
    """

    def __init__(self, num_classes: int = 10, input_channels: int = 1) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # C1: Conv
            nn.Conv2d(input_channels, 6, kernel_size=5),
            nn.ReLU(inplace=True),
            # S2: AvgPool
            nn.AvgPool2d(kernel_size=2, stride=2),
            # C3: Conv
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            # S4: AvgPool
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        return self.classifier(x)


class VGGBlock(nn.Module):
    """
    A VGG-style block with multiple convolutions followed by pooling.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_convs: int = 2,
    ) -> None:
        super().__init__()

        layers = []
        for _ in range(num_convs):
            layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            )
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class SimpleVGG(nn.Module):
    """
    A simplified VGG-style network.
    """

    def __init__(
        self,
        num_classes: int = 10,
        input_channels: int = 3,
    ) -> None:
        super().__init__()

        self.features = nn.Sequential(
            VGGBlock(input_channels, 64, num_convs=2),
            VGGBlock(64, 128, num_convs=2),
            VGGBlock(128, 256, num_convs=3),
            VGGBlock(256, 256, num_convs=3),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    """
    Count the total number of trainable parameters.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device() -> str:
    """
    Get the best available device (CUDA GPU or CPU).

    Returns:
        Device string ('cuda' or 'cpu')
    """
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"
