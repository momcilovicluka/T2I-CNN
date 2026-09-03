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
        self._coordinates = None  # pixel coordinate map after fit()
        self._pix_min = None      # train-derived [0,1] scale (set on first transform)
        self._pix_max = None

    def fit(self, X_train, y_train=None):
        """Learn feature-to-pixel coordinate mapping from training data.

        Invalidates any cached pixel scale from a previous fit so a refit on
        new data never reuses stale train-derived statistics.
        """
        self._pix_min = None
        self._pix_max = None
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
        # Store pixel coordinate map for overlap diagnostics
        # TINTO uses _features_mapping (DataFrame with feature, row, column)
        if hasattr(self.model, '_features_mapping'):
            mapping = self.model._features_mapping
            self._coordinates = mapping[['row', 'column']].values
        elif hasattr(self.model, '_features_positions'):
            self._coordinates = self.model._features_positions.copy()
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

        # FIX (audit 2026-09-03): TINTOlib's TINTO does NOT scale features
        # to [0,1] — blurring compresses peaks (breast cancer max=0.30).
        # After ImageNet normalization (mean 0.485), ALL pixels go negative
        # and pretrained ReLU collapses. Normalize to [0,1] using stats
        # cached from the FIRST transform call (training split — run_all.py
        # always transforms train before val/test), then clip.
        if self._pix_min is None:
            self._pix_min = float(images.min())
            self._pix_max = float(images.max())
        rng = self._pix_max - self._pix_min
        if rng > 1e-8:
            images = (images - self._pix_min) / rng
        images = np.clip(images, 0, 1)

        # Add channel dim: (N, 1, H, W)
        return torch.tensor(images).unsqueeze(1).float()

    def get_coordinates(self):
        """Return feature-to-pixel coordinate mapping.

        Returns: np.ndarray of shape (n_features, 2) with (x, y) pixel
        coordinates, or None if not fitted.
        """
        return self._coordinates
