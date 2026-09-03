"""
Shallow CNN: simple 3-layer convolutional neural network built from scratch.

Architecture:
    Input (1, 32, 32)
    -> Conv2d(1, 32, 3) -> BN -> ReLU -> MaxPool
    -> Conv2d(32, 64, 3) -> BN -> ReLU -> MaxPool
    -> Conv2d(64, 128, 3) -> BN -> ReLU -> AdaptiveAvgPool(4)
    -> Flatten -> Linear(128*4*4, 256) -> ReLU -> Dropout(0.3) -> Linear(256, num_classes)

~620K parameters (measured 618,178). This is the "fair" baseline — no pretrained features,
no architectural advantages. If DeepInsight outperforms naive on this
model, it proves the T2I method matters, not the CNN capacity.

WHY this architecture:
- 3 conv layers are enough to learn local patterns in 32x32 images
- AdaptiveAvgPool(4) handles any input size (32->4, 64->4, etc.)
- Dropout(0.3) prevents overfitting on small datasets
- No residual connections — forces the network to learn direct mappings
"""

import torch.nn as nn
from torchvision import models


class ShallowCNN(nn.Module):
    def __init__(self, num_classes=2, input_channels=1, input_size=32):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: (C, 32, 32) -> (32, 16, 16)
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: (32, 16, 16) -> (64, 8, 8)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: (64, 8, 8) -> (128, 4, 4)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
