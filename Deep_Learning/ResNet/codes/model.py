"""
ResNet Model Definition
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.model(x)
