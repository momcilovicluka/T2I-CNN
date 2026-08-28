"""
Tabular-to-Image transformation methods.

Usage:
    from src.t2i import T2ITransformer
    t = T2ITransformer(method='deepinsight', image_size=32)
    t.fit(X_train, y_train)
    images = t.transform(X_train, y_train)  # shape: (N, 1, 32, 32)
"""

import numpy as np
import os

from .naive import NaiveReshape
from .deepinsight import DeepInsight
from .igtd import IGTD


def _load_tinto_images(temp_dir, N, y):
    """Load images from TINTOlib output directory by sample index.

    TINTOlib stores images as: {class_subfolder}/{zero_padded_index}.npy
    where class_subfolder = str(int(label)).zfill(2)
    and index = original row position in the input DataFrame.

    This function loads by constructing the filename from the sample index,
    ensuring correct order even when input data is shuffled.
    """
    images = []
    for i in range(N):
        label = int(y[i]) if y is not None else 0
        subfolder = str(label).zfill(2)
        filename = str(i).zfill(6) + '.npy'
        img_path = os.path.join(temp_dir, subfolder, filename)
        arr = np.load(img_path)
        images.append(arr)
    return np.stack(images)


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

    def fit(self, X_train, y_train=None):
        """Fit on training data. y_train needed for TINTOlib methods."""
        self.transformer.fit(X_train, y_train)
        return self

    def transform(self, X, y=None):
        """Transform to images. y needed for TINTOlib methods.
        Returns: torch.Tensor of shape (N, 1, H, W)"""
        return self.transformer.transform(X, y)


def verify_all_transformers():
    """Verify all T2I transformers work with breast cancer dataset."""
    from src.preprocessing import preprocess_dataset

    data = preprocess_dataset('breast_cancer')
    X_train = data['X_train']
    y_train = data['y_train']

    for method in ['naive', 'deepinsight', 'igtd']:
        print(f"Testing {method}...")
        t = T2ITransformer(method=method, image_size=32)
        t.fit(X_train, y_train)
        images = t.transform(X_train, y_train)
        print(f"  Shape: {images.shape}")
        print(f"  Pixel range: [{images.min():.3f}, {images.max():.3f}]")
        assert images.shape == (X_train.shape[0], 1, 32, 32), \
            f"Expected ({X_train.shape[0]}, 1, 32, 32), got {images.shape}"
        print(f"  PASS")
