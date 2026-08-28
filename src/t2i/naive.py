"""
Naive Reshape: simplest tabular-to-image transformation.

Takes a feature vector [x1, x2, ..., xn], pads to next perfect square,
and reshapes to a single-channel grayscale image.
"""

import numpy as np
import torch
from PIL import Image


class NaiveReshape:
    def __init__(self, image_size=32):
        self.image_size = image_size
        self.grid_size = None
        self.padded_size = None

    def fit(self, X_train, y_train=None):
        """Compute grid size and normalization stats from training data.

        FIX (Bug #1): Previously, normalization was computed per-call in
        transform(), causing data leakage — each split (train/val/test)
        had different pixel scales. Now we store train min/max here and
        apply them consistently in transform().
        """
        n_features = X_train.shape[1]
        self.grid_size = int(np.ceil(np.sqrt(n_features)))
        self.padded_size = self.grid_size ** 2

        # Compute normalization stats from training data only (no leakage)
        padded_train = np.zeros((X_train.shape[0], self.padded_size), dtype=np.float32)
        padded_train[:, :X_train.shape[1]] = X_train
        grid_train = padded_train.reshape(X_train.shape[0], self.grid_size, self.grid_size)
        self._train_min = grid_train.min()
        self._train_max = grid_train.max()
        return self

    def transform(self, X, y=None):
        """Transform feature vectors to image tensors of shape (N, 1, H, W)."""
        N = X.shape[0]

        # Pad each sample to perfect square length
        padded = np.zeros((N, self.padded_size), dtype=np.float32)
        padded[:, :X.shape[1]] = X

        # Reshape to square grid (N, grid, grid)
        images = padded.reshape(N, self.grid_size, self.grid_size)

        # Resize to target image_size using bicubic interpolation.
        # WHY BICUBIC over NEAREST: Literature (Concern 9) shows bicubic
        # computes weighted averages over 4x4 pixel grids, producing smoother
        # gradients and preserving edge details better when upscaling sparse
        # matrices. Nearest-neighbor creates blocky artifacts that add noise
        # without information. Bicubic is standard for CNN input preparation.
        if self.grid_size != self.image_size:
            resized = np.zeros((N, self.image_size, self.image_size), dtype=np.float32)
            for i in range(N):
                img = Image.fromarray(images[i])
                img = img.resize((self.image_size, self.image_size), Image.BICUBIC)
                resized[i] = np.array(img, dtype=np.float32)
            images = resized

        # Normalize using training statistics (no data leakage).
        # FIX (Bug #1): Previously used images.max() per-call, which gave
        # different scales on train/val/test. Now uses fixed train min/max.
        rng = self._train_max - self._train_min
        if rng > 0:
            images = (images - self._train_min) / rng
        else:
            images = np.zeros_like(images)

        # Clamp to [0, 1] for consistency.
        # WHY: Out-of-distribution test samples can have values outside
        # the training range. Clipping ensures all methods produce [0, 1]
        # pixel values, making CNN training stable across splits.
        images = np.clip(images, 0, 1)

        # Add channel dimension: (N, 1, H, W)
        return torch.tensor(images).unsqueeze(1).float()
