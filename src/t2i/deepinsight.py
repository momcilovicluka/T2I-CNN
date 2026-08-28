"""
DeepInsight: manifold projection-based tabular-to-image transformation.

Uses t-SNE to map feature similarities to 2D pixel coordinates,
then maps feature intensities to pixel locations.

Wraps TINTOlib's DeepInsight implementation.
"""

import numpy as np
import torch
import tempfile
import shutil


class DeepInsight:
    def __init__(self, image_size=32):
        self.image_size = image_size
        self.model = None
        self._temp_dir = None

    def fit(self, X_train, y_train=None):
        """Learn feature-to-pixel coordinate mapping from training data."""
        from TINTOlib.deepInsight import DeepInsight as TINTO_DeepInsight
        import pandas as pd

        df = pd.DataFrame(X_train)
        df['target'] = y_train if y_train is not None else 0

        self.model = TINTO_DeepInsight(
            image_dim=self.image_size,
            problem='classification',
            verbose=False,
            random_seed=42,
            format='npy',
        )
        self.model.fit(df)
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

        # TINTOlib outputs [0, 1] via internal MinMaxScaler.
        # Clamp for out-of-distribution test samples.
        images = np.clip(images, 0, 1)

        # Add channel dim: (N, 1, H, W)
        return torch.tensor(images).unsqueeze(1).float()
