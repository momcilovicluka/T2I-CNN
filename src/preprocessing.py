"""
Dataset loading and preprocessing pipeline.

Handles:
- Loading Breast Cancer Wisconsin, Dry Bean, Adult Income datasets
- Categorical encoding (one-hot for Adult)
- Stratified train/val/test splits
- StandardScaler normalization (fit on train only)
"""


def load_breast_cancer():
    """Load Breast Cancer Wisconsin dataset from sklearn."""
    raise NotImplementedError


def load_dry_bean():
    """Load Dry Bean dataset from UCI."""
    raise NotImplementedError


def load_adult_income():
    """Load Adult Income dataset from UCI."""
    raise NotImplementedError


def preprocess(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """
    Stratified split + StandardScaler normalization.

    Returns: X_train, X_val, X_test, y_train, y_val, y_test
    """
    raise NotImplementedError


def preprocess_dataset(name):
    """
    End-to-end: load + preprocess a dataset by name.

    Args:
        name: 'breast_cancer', 'dry_bean', or 'adult_income'

    Returns:
        dict with keys: X_train, X_val, X_test, y_train, y_val, y_test,
                        feature_names, num_classes
    """
    raise NotImplementedError


def download_datasets():
    """Download all datasets to data/ directory."""
    raise NotImplementedError


def verify_all():
    """Verify all datasets load and preprocess correctly."""
    raise NotImplementedError
