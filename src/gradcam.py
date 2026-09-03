"""
Grad-CAM visualization for T2I images.

Generates heatmaps showing which pixels the CNN focuses on when
making predictions. This directly answers: "Does the CNN actually
use the spatial structure created by T2I methods?"

NOTE: Grad-CAM works best with ShallowCNN (4x4 feature maps at 32x32).
ResNet-18 on 32x32 produces only 2x2 maps at layer4 — we use layer3
instead (4x4 maps), but the resolution is still limited. For the paper,
Grad-CAM results for ShallowCNN are most interpretable.

Usage:
    from src.gradcam import generate_gradcam, overlay_heatmap
    heatmap = generate_gradcam(model, image, target_class, arch='shallow')
"""

import numpy as np
import torch
import torch.nn as nn


def get_target_layer(model, arch):
    """Get the target convolutional layer for Grad-CAM.

    WHY: Grad-CAM needs a convolutional layer with spatial output.
    The last conv layer captures the most class-discriminative features
    while still retaining spatial information.

    ShallowCNN: features[8] is the last Conv2d (128 channels, 4x4 spatial)
    ResNet-18: layer3[-1].conv2 (256 channels, 4x4 spatial at 32x32 input)
    """
    if arch == 'shallow':
        return model.features[8]
    elif arch in ('resnet', 'resnet_scratch'):
        # Use layer3 instead of layer4 — layer4 produces 2x2 maps on 32x32
        # layer3 produces 4x4 maps which are still small but better
        return model.backbone.layer3[-1].conv2
    else:
        raise ValueError(f"Grad-CAM not supported for architecture: {arch}")


def generate_gradcam(model, image, target_class, arch, device='cpu'):
    """Generate Grad-CAM heatmap for a single image.

    Args:
        model: trained CNN model
        image: torch.Tensor of shape (1, 1, H, W) — single grayscale image
        target_class: int — class index to generate heatmap for
        arch: str — 'shallow', 'resnet', or 'resnet_scratch'
        device: str — 'cpu' or 'cuda'

    Returns:
        heatmap: np.ndarray of shape (H, W) — values in [0, 1]
    """
    from pytorch_grad_cam import GradCAM

    from src.train import imagenet_normalize

    model = model.to(device)
    model.eval()
    use_imagenet_norm = getattr(model, 'pretrained', False)

    target_layer = get_target_layer(model, arch)
    cam = GradCAM(model=model, target_layers=[target_layer])

    image_tensor = image.to(device)
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    if use_imagenet_norm:
        image_tensor = imagenet_normalize(image_tensor)

    orig_h, orig_w = image_tensor.shape[2], image_tensor.shape[3]

    grayscale_cam = cam(input_tensor=image_tensor, targets=None)
    heatmap = grayscale_cam[0]

    # Downscale back to original size if needed
    if heatmap.shape[0] != orig_h or heatmap.shape[1] != orig_w:
        heatmap_t = torch.tensor(heatmap).unsqueeze(0).unsqueeze(0).float()
        heatmap_t = torch.nn.functional.interpolate(
            heatmap_t, size=(orig_h, orig_w), mode='bilinear', align_corners=False
        )
        heatmap = heatmap_t.squeeze().numpy()

    # Normalize to [0, 1]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    return heatmap


def overlay_heatmap(image, heatmap, alpha=0.5):
    """Overlay a Grad-CAM heatmap on an image.

    Args:
        image: np.ndarray of shape (H, W) — grayscale image in [0, 1]
        heatmap: np.ndarray of shape (H, W) — heatmap in [0, 1]
        alpha: float — blending factor

    Returns:
        overlay: np.ndarray of shape (H, W, 3) — RGB image with heatmap
    """
    import matplotlib.cm as cm

    # Convert grayscale to RGB
    if image.ndim == 2:
        image_rgb = np.stack([image] * 3, axis=-1)
    elif image.ndim == 3 and image.shape[0] == 1:
        image_rgb = np.stack([image[0]] * 3, axis=-1)
    else:
        image_rgb = image.copy()

    # Apply colormap to heatmap
    colormap = cm.get_cmap('jet')
    heatmap_colored = colormap(heatmap)[:, :, :3]  # Drop alpha channel

    # Blend
    overlay = (1 - alpha) * image_rgb + alpha * heatmap_colored
    overlay = np.clip(overlay, 0, 1)

    return overlay
