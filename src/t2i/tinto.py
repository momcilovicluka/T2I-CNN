"""
TINTO: Tabular data to Image transformation with blurring.

Similar to DeepInsight but adds an artistic blurring filter to smooth
spatial patterns, reducing sharp transitions between adjacent features.
This creates continuous gradients that CNNs can extract features from
more effectively.

Key difference from DeepInsight:
- DeepInsight: t-SNE -> pixel coordinates, no smoothing
- TINTO: PCA/t-SNE -> pixel coordinates + Gaussian blurring filter

The blurring propagates signal to adjacent pixels with decaying intensity,
creating smoother spatial patterns. This helps CNN kernels (3x3) capture
meaningful local relationships even when features are sparse.

Wraps TINTOlib's TINTO implementation.
"""

import numpy as np
import torch
import tempfile
import shutil


class TINTO:
    def __init__(self, image_size=32):
        self.image_size = image_size
        self.model = None
        self._temp_dir = None

    def fit(self, X_train, y_train=None):
        """Learn feature-to-pixel coordinate mapping from training data."""
        from TINTOlib.tinto import TINTO as TINTO_TINTO
        import pandas as pd

        df = pd.DataFrame(X_train)
        df['target'] = y_train if y_train is not None else 0

        self.model = TINTO_TINTO(
            problem='classification',
            verbose=False,
            pixels=self.image_size,
            algorithm='PCA',
            blur=True,          # Enable blurring — the key TINTO feature
            submatrix=True,
            amplification=3.14,
            distance=2,
            steps=4,
            option='mean',
            times=4,
            zoom=1,
            format='npy',
            cmap='binary',
            random_seed=42,
        )
        self.model.fit(df)
        return self

    def transform(self, X, y=None):
        """Transform feature vectors to image tensors of shape (N, 1, H, W).

        TINTO outputs blurred images where feature values are propagated
        to neighboring pixels. Like DeepInsight, outputs [0, 1] range
        via internal MinMaxScaler.
        """
        from . import _load_tinto_images

        N = X.shape[0]

        # Create temp dir and run TINTOlib transform
        self._temp_dir = tempfile.mkdtemp()

        import pandas as pd
        df = pd.DataFrame(X)
        df['target'] = y if y is not None else 0
        self.model.transform(df, self._temp_dir)

        # Load images by index (correct order even if input is shuffled)
        images = _load_tinto_images(self._temp_dir, N, y)

        # Cleanup
        shutil.rmtree(self._temp_dir, ignore_errors=True)
        self._temp_dir = None

        # Clamp to [0, 1] for out-of-distribution test samples
        images = np.clip(images, 0, 1)

        # Add channel dim: (N, 1, H, W)
        return torch.tensor(images).unsqueeze(1).float()
