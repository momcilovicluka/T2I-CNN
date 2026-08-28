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
        """Compute grid size from number of features (no data leakage)."""
        n_features = X_train.shape[1]
        self.grid_size = int(np.ceil(np.sqrt(n_features)))
        self.padded_size = self.grid_size ** 2
        return self

    def transform(self, X, y=None):
        """Transform feature vectors to image tensors of shape (N, 1, H, W)."""
        N = X.shape[0]
        
        # Pad each sample to perfect square length
        padded = np.zeros((N, self.padded_size), dtype=np.float32)
        padded[:, :X.shape[1]] = X
        
        # Reshape to square grid (N, grid, grid)
        images = padded.reshape(N, self.grid_size, self.grid_size)
        
        # Resize to target image_size using nearest neighbor
        if self.grid_size != self.image_size:
            resized = np.zeros((N, self.image_size, self.image_size), dtype=np.float32)
            for i in range(N):
                img = Image.fromarray(images[i])
                img = img.resize((self.image_size, self.image_size), Image.NEAREST)
                resized[i] = np.array(img, dtype=np.float32)
            images = resized
        
        # Normalize to [0, 1] range
        if images.max() > 0:
            images = images / images.max()
        
        # Add channel dimension: (N, 1, H, W)
        return torch.tensor(images).unsqueeze(1).float()
