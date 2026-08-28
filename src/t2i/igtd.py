"""
IGTD: Image Generator for Tabular Data.

Rank-based permutation method that matches feature distance rankings
with pixel distance rankings via Frobenius norm minimization.

Wraps TINTOlib's IGTD implementation.
"""

import numpy as np
import torch
import tempfile
import shutil


# IGTD internally outputs images in [0, 255] range.
# We normalize to [0, 1] by dividing by 255.
IGTD_RAW_MAX = 255.0


class IGTD:
    def __init__(self, image_size=32):
        self.image_size = image_size
        self.model = None
        self._temp_dir = None

    def fit(self, X_train, y_train=None):
        """Learn feature-to-pixel coordinate mapping from training data."""
        from TINTOlib.igtd import IGTD as TINTO_IGTD
        import pandas as pd

        df = pd.DataFrame(X_train)
        df['target'] = y_train if y_train is not None else 0

        self.model = TINTO_IGTD(
            scale=[self.image_size, self.image_size],
            problem='classification',
            verbose=False,
            random_seed=42,
            format='npy',
            fea_dist_method='Pearson',
            image_dist_method='Euclidean',
            error='squared',
            max_step=1000,
            val_step=50,
        )
        self.model.fit(df)
        # No need to run transform here — IGTD output range is known [0, 255]
        return self

    def transform(self, X, y=None):
        """Transform feature vectors to image tensors of shape (N, 1, H, W)."""
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

        # IGTD outputs [0, 255]. Normalize to [0, 1].
        images = images / IGTD_RAW_MAX

        # Clamp for consistency
        images = np.clip(images, 0, 1)

        # Add channel dim: (N, 1, H, W)
        return torch.tensor(images).unsqueeze(1).float()
