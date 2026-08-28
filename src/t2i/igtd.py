"""
IGTD: Image Generator for Tabular Data.

Rank-based permutation method that matches feature distance rankings
with pixel distance rankings via Frobenius norm minimization.
"""


class IGTD:
    def __init__(self, image_size=32):
        self.image_size = image_size

    def fit(self, X_train):
        """Learn feature-to-pixel coordinate mapping from training data."""
        raise NotImplementedError

    def transform(self, X):
        """Transform feature vectors to image tensors of shape (N, 1, H, W)."""
        raise NotImplementedError
