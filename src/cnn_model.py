"""SimpleCNN architecture for 2D-encoded time-series classification."""

import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    Compact CNN for single-channel 2D image classification.

    Input:  (batch, 1, H, W)
    Block1: Conv2d(1,  32, 3, pad=1) -> BN -> ReLU -> MaxPool(2)
    Block2: Conv2d(32, 64, 3, pad=1) -> BN -> ReLU -> MaxPool(2)
    Block3: Conv2d(64,128, 3, pad=1) -> BN -> ReLU -> AdaptiveAvgPool(4)
    FC:     Linear(2048,128) -> ReLU -> Dropout(0.3) -> Linear(128, n_classes)
    """

    def __init__(self, n_classes: int):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)
