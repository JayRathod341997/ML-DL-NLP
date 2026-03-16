"""
Neural Network Model Implementation

A modular PyTorch implementation of a Multi-Layer Perceptron (MLP)
for classification tasks. Uses device-agnostic setup for CPU/GPU.
"""

from torch import nn, Tensor
from typing import Sequence


class NeuralNetwork(nn.Module):
    """
    A feedforward neural network (Multi-Layer Perceptron) for classification.

    Architecture:
        Input -> Hidden Layers (with ReLU) -> Output (with Softmax)

    Attributes:
        layers: Sequential container of linear layers and activations

    Example:
        >>> model = NeuralNetwork(input_size=784, hidden_sizes=[256, 128], num_classes=10)
        >>> x = torch.randn(32, 784)
        >>> output = model(x)  # Shape: (32, 10)
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int] | None = None,
        num_classes: int = 10,
        dropout_rate: float = 0.2,
    ) -> None:
        """
        Initialize the neural network.

        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes. Defaults to [256, 128]
            num_classes: Number of output classes
            dropout_rate: Dropout probability for regularization
        """
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [256, 128]

        # Build network layers
        layers: list[nn.Module] = []

        # First layer: input -> first hidden
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Flatten input if it's an image (batch_size, 1, height, width)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        return self.layers(x)

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


class RegressionNetwork(nn.Module):
    """
    A feedforward neural network for regression tasks.

    Architecture:
        Input -> Hidden Layers (with ReLU) -> Output (linear)
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int] | None = None,
        output_size: int = 1,
        dropout_rate: float = 0.2,
    ) -> None:
        """
        Initialize the regression network.

        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            output_size: Number of output targets
            dropout_rate: Dropout probability
        """
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [128, 64]

        layers: list[nn.Module] = []

        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        # Output layer (no activation for regression)
        layers.append(nn.Linear(prev_size, output_size))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for regression.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Predicted values of shape (batch_size, output_size)
        """
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        return self.layers(x)


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
