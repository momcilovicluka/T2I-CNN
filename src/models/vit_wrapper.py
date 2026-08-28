"""
Vision Transformer (ViT) wrapper for tabular-to-image classification.

Uses timm library to load pretrained ViT, adapts for grayscale input.

WHY ViT:
- Self-attention mechanism captures global relationships between features
- Contrasts with CNN's local receptive fields
- State-of-the-art on many vision benchmarks
- Important to include for comprehensive comparison

WHY resize to 224x224:
- ViT-Base uses patch_size=16, so 32x32 = 4 patches → too few for attention
- 224x224 = 196 patches → rich attention patterns
- Bilinear resize preferred over nearest-neighbor (smoother gradients)
- Resize from 32→224 creates interpolated images, which is acceptable
  for comparing T2I methods (all methods equally affected)

Usage:
    model = ViTWrapper(num_classes=2, pretrained=True)
    # Input: (N, 1, 224, 224) — resize images before passing
"""

import torch
import torch.nn as nn
try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


class ViTWrapper(nn.Module):
    def __init__(self, num_classes=2, input_channels=1, pretrained=True,
                 img_size=224, patch_size=16):
        super().__init__()

        if not HAS_TIMM:
            raise ImportError(
                "timm is required for ViT. Install with: pip install timm"
            )

        # Load pretrained ViT-Base
        # vit_base_patch16_224: 86M params, patch_size=16, img_size=224
        self.vit = timm.create_model(
            'vit_base_patch16_224',
            pretrained=pretrained,
            num_classes=num_classes,
            img_size=img_size,
        )

        # Adapt patch embedding for grayscale input if needed
        if input_channels != 3:
            orig_proj = self.vit.patch_embed.proj
            self.vit.patch_embed.proj = nn.Conv2d(
                input_channels,
                orig_proj.out_channels,
                kernel_size=orig_proj.kernel_size,
                stride=orig_proj.stride,
                padding=orig_proj.padding,
                bias=orig_proj.bias is not None,
            )
            if pretrained:
                with torch.no_grad():
                    if input_channels == 1:
                        self.vit.patch_embed.proj.weight = nn.Parameter(
                            orig_proj.weight.mean(dim=1, keepdim=True)
                        )
                    else:
                        nn.init.kaiming_normal_(self.vit.patch_embed.proj.weight)

        self.pretrained = pretrained
        self.img_size = img_size
        self.patch_size = patch_size

    def forward(self, x):
        # ViT expects (N, C, H, W) — resize if needed
        if x.shape[-1] != self.img_size or x.shape[-2] != self.img_size:
            x = torch.nn.functional.interpolate(
                x, size=(self.img_size, self.img_size),
                mode='bilinear', align_corners=False,
            )
        return self.vit(x)

    def get_param_groups(self, lr_backbone=1e-5, lr_head=1e-3):
        """Parameter groups: low LR for pretrained, high for new head."""
        if self.pretrained:
            backbone_params = []
            head_params = []
            for name, param in self.vit.named_parameters():
                if 'head' in name or 'fc' in name:
                    head_params.append(param)
                else:
                    backbone_params.append(param)
            return [
                {'params': backbone_params, 'lr': lr_backbone},
                {'params': head_params, 'lr': lr_head},
            ]
        else:
            return [{'params': self.vit.parameters(), 'lr': lr_head}]

    def freeze_backbone(self):
        """Freeze all except classification head. For LP-FT Phase 1."""
        for name, param in self.vit.named_parameters():
            if 'head' not in name and 'fc' not in name:
                param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all. For LP-FT Phase 2."""
        for param in self.vit.parameters():
            param.requires_grad = True
