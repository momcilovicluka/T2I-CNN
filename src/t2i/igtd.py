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
import os
import pandas as pd


class IGTD:
    def __init__(self, image_size=32):
        self.image_size = image_size
        self.model = None
        self._temp_dir = None

    def fit(self, X_train, y_train=None):
        """Learn feature-to-pixel coordinate mapping from training data."""
        from TINTOlib.igtd import IGTD as TINTO_IGTD

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

        # Compute normalization stats from training images
        # IGTD outputs [0, 255], we need consistent [0, 1]
        tmp = tempfile.mkdtemp()
        self.model.transform(df, tmp)
        cls = pd.read_csv(os.path.join(tmp, 'classification.csv'))
        train_min, train_max = float('inf'), float('-inf')
        for _, row in cls.iterrows():
            arr = np.load(os.path.join(tmp, row['images']))
            train_min = min(train_min, arr.min())
            train_max = max(train_max, arr.max())
        self._train_min = train_min
        self._train_max = train_max
        shutil.rmtree(tmp, ignore_errors=True)
        return self

    def transform(self, X, y=None):
        """Transform feature vectors to image tensors of shape (N, 1, H, W)."""
        df = pd.DataFrame(X)
        df['target'] = y if y is not None else 0

        self._temp_dir = tempfile.mkdtemp()
        self.model.transform(df, self._temp_dir)

        # Load images in order from classification.csv
        cls_path = os.path.join(self._temp_dir, 'classification.csv')
        cls = pd.read_csv(cls_path)

        images = []
        for _, row in cls.iterrows():
            img_path = os.path.join(self._temp_dir, row['images'])
            arr = np.load(img_path)
            images.append(arr)

        images = np.stack(images)  # (N, H, W)

        # Cleanup
        shutil.rmtree(self._temp_dir, ignore_errors=True)
        self._temp_dir = None

        # IGTD outputs [0, 255]. Normalize to [0, 1] using training stats.
        rng = self._train_max - self._train_min
        if rng > 0:
            images = (images - self._train_min) / rng

        # Clamp to [0, 1] for consistency
        images = np.clip(images, 0, 1)

        # Add channel dim: (N, 1, H, W)
        return torch.tensor(images).unsqueeze(1).float()
