"""
Naive Reshape: simplest tabular-to-image transformation.

Takes a feature vector [x1, x2, ..., xn], pads to next perfect square,
and reshapes to a single-channel grayscale image.
"""


class NaiveReshape:
    def __init__(self, image_size=32):
        self.image_size = image_size

    def fit(self, X_train):
        """Fit on training data (no-op for naive reshape)."""
        raise NotImplementedError

    def transform(self, X):
        """Transform feature vectors to image tensors of shape (N, 1, H, W)."""
        raise NotImplementedError
