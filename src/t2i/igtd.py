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


# IGTD internally outputs images in [0, 255] range (matplotlib colormap).
# DeepInsight outputs [0, 1] (MinMaxScaler on features).
# FIX (Bug #3): These ranges were mismatched — both used images.max()
# which happened to work for IGTD (~255/255=1) but was fragile.
# Now IGTD explicitly divides by 255.0 for deterministic normalization.
IGTD_RAW_MAX = 255.0


class IGTD:
    def __init__(self, image_size=32):
        self.image_size = image_size
        self.model = None
        self._temp_dir = None
        self._coordinates = None

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
        # FIX (Bug #5): Previously ran self.model.transform() here just to
        # compute min/max for normalization. This doubled IGTD fit time.
        # Now we use the known output range [0, 255] directly.
        # Store pixel coordinate map for overlap diagnostics
        if hasattr(self.model, '_features_mapping'):
            mapping = self.model._features_mapping
            self._coordinates = mapping[['row', 'column']].values
        elif hasattr(self.model, '_features_positions'):
            self._coordinates = self.model._features_positions.copy()
        return self

    def transform(self, X, y=None):
        """Transform feature vectors to image tensors of shape (N, 1, H, W).

        FIX (Bug #3): IGTD outputs [0, 255] (matplotlib colormap), while
        DeepInsight outputs [0, 1] (MinMaxScaler). Normalizes by /255.0
        to match DeepInsight's range.

        FIX (Bug #4): Uses _load_tinto_images() for index-based loading.
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

        # IGTD outputs [0, 255]. Normalize to [0, 1] for consistency
        # with DeepInsight which outputs [0, 1] via MinMaxScaler.
        images = images / IGTD_RAW_MAX

        # Clamp for out-of-distribution test samples
        images = np.clip(images, 0, 1)

        # Add channel dim: (N, 1, H, W)
        return torch.tensor(images).unsqueeze(1).float()

    def get_coordinates(self):
        """Return feature-to-pixel coordinate mapping.

        IGTD is collision-free by design, so OF/OP will always be 0.
        Coordinates are useful for visualizing the layout.
        """
        return self._coordinates
