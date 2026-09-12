"""
CNN model architectures for tabular-to-image classification.
"""

from .shallow_cnn import ShallowCNN
from .resnet_wrapper import ResNetWrapper
from .vit_wrapper import ViTWrapper


def get_model(name, num_classes=2, **kwargs):
    """Factory function to get a model by name.

    Audit C14: 'resnet_scratch' is included so this public factory matches the
    architectures the study actually runs. It defaults to random init and
    1-channel input — the same configuration run_all.py uses (see
    run_all.create_cnn_model for the documented pretrained-vs-scratch input
    confound).
    """
    if name == 'resnet_scratch':
        kwargs.setdefault('pretrained', False)
        kwargs.setdefault('input_channels', 1)
    models = {
        'shallow': ShallowCNN,
        'resnet': ResNetWrapper,
        'resnet_scratch': ResNetWrapper,
        'vit': ViTWrapper,
    }
    if name not in models:
        raise ValueError(f"Unknown model: {name}. Choose from {list(models)}")
    return models[name](num_classes=num_classes, **kwargs)


def verify_all_models():
    """Verify all models produce correct output shapes."""
    import torch

    x = torch.randn(4, 1, 32, 32)
    for name in ['shallow', 'resnet', 'resnet_scratch', 'vit']:
        try:
            model = get_model(name, num_classes=2)
            out = model(x)
            params = sum(p.numel() for p in model.parameters())
            print(f"  {name}: output={out.shape}, params={params:,}")
        except NotImplementedError:
            print(f"  {name}: NOT IMPLEMENTED YET")
        except Exception as e:
            print(f"  {name}: ERROR - {e}")
