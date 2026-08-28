"""
Vision Transformer (ViT) transfer learning wrapper.

Uses timm to load pretrained ViT, adapts for grayscale input.
"""

import torch.nn as nn


class ViTWrapper(nn.Module):
    def __init__(self, num_classes=2, input_channels=1, pretrained=True):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def get_param_groups(self, lr_backbone=1e-4, lr_head=1e-3):
        """Return parameter groups with different learning rates for fine-tuning."""
        raise NotImplementedError
