"""
U-Net Model Definition
"""

import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        # Simplified U-Net
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3), nn.ReLU(), nn.Conv2d(64, 64, 3), nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3), nn.ReLU(), nn.Conv2d(128, 128, 3), nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3), nn.ReLU(), nn.Conv2d(64, 64, 3), nn.ReLU()
        )
        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        # Simplified forward
        return self.final(self.dec2(self.enc2(self.enc1(x))))
