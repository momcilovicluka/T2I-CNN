"""
ResNet-18 wrapper for tabular-to-image classification.

Supports two modes:
- pretrained=True: ImageNet pretrained weights (transfer learning)
- pretrained=False: Random initialization (capacity control)

WHY ResNet-18:
- Standard architecture for image classification benchmarks
- Residual connections enable training deeper networks
- Well-studied in transfer learning literature
- ~11M parameters (55x more than ShallowCNN)

WHY from-scratch option:
- Without it, ResNet advantages could come from pretrained features,
  not from the T2I method quality
- Training from scratch on synthetic images isolates the T2I method
  effect from the architecture capacity effect
- Paper should report both to discuss transfer learning effectiveness

LP-FT (Linear Probing then Fine-Tuning):
- Phase 1: Freeze backbone, train only the new FC head
- Phase 2: Unfreeze all layers, train end-to-end with low LR
- This two-phase strategy stabilizes training when the model
  is much larger than the dataset (e.g., 11M params on 398 samples)
"""

import torch
import torch.nn as nn
from torchvision import models


class ResNetWrapper(nn.Module):
    def __init__(self, num_classes=2, input_channels=1, pretrained=True):
        super().__init__()

        # Load ResNet-18
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None

        self.backbone = models.resnet18(weights=weights)

        # Adapt first conv layer for different input channels
        # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # For grayscale: Conv2d(1, 64, ...)
        # For RGB repeated: Conv2d(3, 64, ...) — no change needed
        if input_channels != 3:
            orig_conv = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                input_channels, 64,
                kernel_size=orig_conv.kernel_size,
                stride=orig_conv.stride,
                padding=orig_conv.padding,
                bias=orig_conv.bias is not None,
            )
            # If pretrained, copy weights by averaging across input channels
            if pretrained:
                with torch.no_grad():
                    if input_channels == 1:
                        # Average RGB weights -> grayscale
                        self.backbone.conv1.weight = nn.Parameter(
                            orig_conv.weight.mean(dim=1, keepdim=True)
                        )
                    else:
                        # Random init for non-standard channel counts
                        nn.init.kaiming_normal_(self.backbone.conv1.weight)

        # Replace final FC layer
        in_features = self.backbone.fc.in_features  # 512
        self.backbone.fc = nn.Linear(in_features, num_classes)

        self.pretrained = pretrained

    def forward(self, x):
        return self.backbone(x)

    def get_param_groups(self, lr_backbone=1e-4, lr_head=1e-3):
        """Return parameter groups with different learning rates.

        WHY: Pretrained backbone needs low LR to avoid destroying
        learned features. New FC head needs higher LR to learn fast.

        For from-scratch mode, all layers use lr_head since there's
        no pretrained knowledge to preserve.
        """
        if self.pretrained:
            # Exclude the final FC layer
            backbone_params = []
            head_params = []
            for name, param in self.backbone.named_parameters():
                if 'fc' in name:
                    head_params.append(param)
                else:
                    backbone_params.append(param)
            return [
                {'params': backbone_params, 'lr': lr_backbone},
                {'params': head_params, 'lr': lr_head},
            ]
        else:
            # From scratch — single LR for all
            return [{'params': self.backbone.parameters(), 'lr': lr_head}]

    def freeze_backbone(self):
        """Freeze all layers except the final FC. For LP-FT Phase 1."""
        for name, param in self.backbone.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all layers. For LP-FT Phase 2."""
        for param in self.backbone.parameters():
            param.requires_grad = True
