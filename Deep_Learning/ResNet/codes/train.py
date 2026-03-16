"""
ResNet Training Script
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from model import ResNetClassifier


def train():
    transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
    dataset = datasets.MNIST("./data", train=True, transform=transform, download=True)

    model = ResNetClassifier(num_classes=10)
    print("Training ResNet model...")


if __name__ == "__main__":
    train()
