"""
DeepInsight: manifold projection-based tabular-to-image transformation.

Uses t-SNE or PCA to map feature similarities to 2D pixel coordinates,
then maps feature intensities to pixel locations.
"""


class DeepInsight:
    def __init__(self, image_size=32, method='tsne'):
        self.image_size = image_size
        self.method = method

    def fit(self, X_train):
        """Learn feature-to-pixel coordinate mapping from training data."""
        raise NotImplementedError

    def transform(self, X):
        """Transform feature vectors to image tensors of shape (N, 1, H, W)."""
        raise NotImplementedError
