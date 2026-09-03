"""
DeepInsight: manifold projection-based tabular-to-image transformation.

Projects features as points in sample space and maps them to 2D pixel
coordinates. The projection algorithm is TINTOlib's default, PCA
(algorithm_rd='pca'); this wrapper does not override it, so the
coordinates reflect the linear (correlation) structure of the features.
Intensities are then placed at the mapped pixel locations.

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
        self._coordinates = None  # pixel coordinate map after fit()

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
        # Store pixel coordinate map for overlap diagnostics
        # WHY (Step 3): OF/OP metrics need the feature->pixel mapping
        if hasattr(self.model, '_features_positions'):
            self._coordinates = self.model._features_positions.copy()
        return self

    def transform(self, X, y=None):
        """Transform feature vectors to image tensors of shape (N, 1, H, W).

        FIX (Bug #2): Removed redundant images/images.max() normalization.
        TINTOlib already outputs [0, 1] via internal MinMaxScaler. The
        extra normalization distorted spatial mapping and created inconsistent
        scales across splits (val max=0.90, test max=1.25 for breast cancer).

        FIX (Bug #4): Now uses _load_tinto_images() which loads by sample
        index instead of CSV row order. TINTOlib's classification.csv lists
        files in sequential filename order, not input DataFrame order.
        Shuffled input would previously assign wrong images to wrong samples.
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

        # Clamp to [0, 1] for out-of-distribution test samples.
        # WHY: TINTOlib's MinMaxScaler is fit on training data, so test
        # values can exceed [0, 1]. Clipping ensures consistent range.
        # NOTE: DeepInsight natively outputs [0, 1] (verified empirically),
        # so no train-derived rescale cache is needed here — only TINTO
        # requires one (see src/t2i/tinto.py).
        images = np.clip(images, 0, 1)

        # Add channel dim: (N, 1, H, W)
        return torch.tensor(images).unsqueeze(1).float()

    def get_coordinates(self):
        """Return feature-to-pixel coordinate mapping.

        WHY (Step 3): Enables overlap diagnostics (OF/OP) without
        re-running the T2I transformation.

        Returns: np.ndarray of shape (n_features, 2) with (x, y) pixel
        coordinates, or None if not fitted.
        """
        return self._coordinates
