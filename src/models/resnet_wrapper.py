"""
ResNet-18 transfer learning wrapper.

Loads pretrained ResNet-18, adapts first conv layer for grayscale input,
replaces final FC layer for target number of classes.
"""

import torch.nn as nn


class ResNetWrapper(nn.Module):
    def __init__(self, num_classes=2, input_channels=1, pretrained=True):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def get_param_groups(self, lr_backbone=1e-4, lr_head=1e-3):
        """Return parameter groups with different learning rates for fine-tuning."""
        raise NotImplementedError
