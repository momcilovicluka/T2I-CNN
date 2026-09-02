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
from .tinto import TINTO
from .deepinsight import DeepInsight
from .igtd import IGTD
from .s_igtd import SIGTD


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

        # Alignment assertion: verify file exists
        if not os.path.exists(img_path):
            raise FileNotFoundError(
                f"TINTOlib image not found: {img_path}"
                f"\n  sample={i}, label={label}, temp_dir={temp_dir}"
                f"\n  Available files: {os.listdir(os.path.join(temp_dir, subfolder)) if os.path.isdir(os.path.join(temp_dir, subfolder)) else 'subfolder missing'}"
            )

        arr = np.load(img_path)
        images.append(arr)
    return np.stack(images)


def compute_optimal_image_size(n_features, min_size=8, max_size=64):
    """Compute image size based on feature count.

    WHY (Concern 8 — 32x32 too small for 108 features):
    Feature density = n_features / image_size^2. With 108 features on
    32x32 = 1024 pixels, density = 10.5%. CNN kernels (3x3) see mostly
    zeros. Increasing to ~17x17 for 108 features gives 37% density,
    making convolution meaningful.

    Strategy: Find the smallest square image where feature density is
    at least 20% (i.e., at least 1 in 5 pixels carries information).
    Minimum size is ceil(sqrt(n_features)) to ensure every feature
    gets at least 1 pixel.

    Args:
        n_features: int, number of input features
        min_size: int, minimum image dimension (default 8)
        max_size: int, maximum image dimension (default 64)

    Returns:
        int, optimal image size (H = W)
    """
    import math
    # Minimum size: every feature gets at least 1 pixel
    min_needed = max(min_size, int(math.ceil(math.sqrt(n_features))))
    min_needed = int(math.ceil(min_needed / 8) * 8)  # round to multiple of 8

    # Target: at least 20% density (1 in 5 pixels is a feature)
    for size in range(min_needed, max_size + 1, 8):
        if n_features / (size * size) >= 0.2:
            return size
    return max_size


class T2ITransformer:
    """Unified interface for all T2I transformation methods."""

    METHODS = {
        'naive': NaiveReshape,
        'tinto': TINTO,
        'deepinsight': DeepInsight,
        'igtd': IGTD,
        's_igtd': SIGTD,
    }

    def __init__(self, method='naive', image_size=32, auto_size=False, **kwargs):
        """Initialize T2I transformer.

        Args:
            method: 'naive', 'deepinsight', or 'igtd'
            image_size: int, output image dimension (H=W)
            auto_size: bool, if True, compute image_size from n_features
                in fit() to ensure sufficient feature density
        """
        if method not in self.METHODS:
            raise ValueError(f"Unknown method: {method}. Choose from {list(self.METHODS)}")
        self.method = method
        self.image_size = image_size
        self.auto_size = auto_size
        self._kwargs = kwargs
        self.transformer = self.METHODS[method](image_size=image_size, **kwargs)

    def fit(self, X_train, y_train=None):
        """Fit on training data. y_train needed for TINTOlib methods.

        If auto_size=True, computes optimal image size from X_train.shape[1]
        and reinitializes the transformer with that size.
        """
        if self.auto_size:
            n_features = X_train.shape[1]
            optimal_size = compute_optimal_image_size(n_features)
            if optimal_size != self.image_size:
                print(f"  Auto-sizing: {self.image_size}x{self.image_size} -> "
                      f"{optimal_size}x{optimal_size} for {n_features} features "
                      f"(density={n_features/optimal_size**2:.1%})")
                self.image_size = optimal_size
                self.transformer = self.METHODS[self.method](
                    image_size=optimal_size, **self._kwargs
                )

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

    for method in ['naive', 'tinto', 'deepinsight', 'igtd', 's_igtd']:
        print(f"Testing {method}...")
        t = T2ITransformer(method=method, image_size=32)
        t.fit(X_train, y_train)
        images = t.transform(X_train, y_train)
        print(f"  Shape: {images.shape}")
        print(f"  Pixel range: [{images.min():.3f}, {images.max():.3f}]")
        assert images.shape == (X_train.shape[0], 1, 32, 32), \
            f"Expected ({X_train.shape[0]}, 1, 32, 32), got {images.shape}"
        print(f"  PASS")
