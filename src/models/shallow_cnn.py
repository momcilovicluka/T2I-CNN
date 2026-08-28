"""
Shallow CNN: simple 3-layer convolutional neural network built from scratch.

Architecture:
    Input (1, 32, 32)
    -> Conv2d(1, 32, 3) -> BN -> ReLU -> MaxPool
    -> Conv2d(32, 64, 3) -> BN -> ReLU -> MaxPool
    -> Conv2d(64, 128, 3) -> BN -> ReLU -> AdaptiveAvgPool(4)
    -> Flatten -> Linear(128*4*4, 256) -> ReLU -> Dropout -> Linear(256, num_classes)
"""

import torch.nn as nn


class ShallowCNN(nn.Module):
    def __init__(self, num_classes=2, input_channels=1, input_size=32):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
