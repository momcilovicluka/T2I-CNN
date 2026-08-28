"""
Tabular-to-Image transformation methods.

Usage:
    from src.t2i import T2ITransformer
    t = T2ITransformer(method='deepinsight', image_size=32)
    t.fit(X_train)
    images = t.transform(X_train)  # shape: (N, 1, 32, 32)
"""

from .naive import NaiveReshape
from .deepinsight import DeepInsight
from .igtd import IGTD


class T2ITransformer:
    """Unified interface for all T2I transformation methods."""

    METHODS = {
        'naive': NaiveReshape,
        'deepinsight': DeepInsight,
        'igtd': IGTD,
    }

    def __init__(self, method='naive', image_size=32, **kwargs):
        if method not in self.METHODS:
            raise ValueError(f"Unknown method: {method}. Choose from {list(self.METHODS)}")
        self.transformer = self.METHODS[method](image_size=image_size, **kwargs)
        self.method = method

    def fit(self, X_train):
        self.transformer.fit(X_train)
        return self

    def transform(self, X):
        return self.transformer.transform(X)


def verify_all_transformers():
    """Verify all T2I transformers work with dummy data."""
    import torch
    from src.preprocessing import preprocess_dataset

    X_dummy = torch.randn(100, 30).numpy()  # 100 samples, 30 features

    for method in ['naive', 'deepinsight', 'igtd']:
        t = T2ITransformer(method=method, image_size=32)
        t.fit(X_dummy)
        images = t.transform(X_dummy)
        assert images.shape == (100, 1, 32, 32), f"{method}: expected (100,1,32,32), got {images.shape}"
        print(f"✓ {method}: {images.shape}")
