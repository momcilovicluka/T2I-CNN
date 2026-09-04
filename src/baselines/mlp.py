"""MLP baseline for tabular data classification."""

import inspect

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.neural_network import MLPClassifier
from src.evaluate import evaluate_tabular


def train_and_evaluate(X_train, y_train, X_test, y_test, num_classes=2):
    """Train MLP and return metrics dict.

    WHY MLP as baseline:
    - Simple neural network on raw tabular features
    - No spatial structure — direct comparison with CNN+T2I
    - If MLP ≈ CNN, the T2I transformation adds no value
    - If CNN > MLP, spatial structure helps
    - Per-sample balanced weights mirror the inverse-frequency class
      weights used in the CNN loss (professor-validation 10.1).
    """
    classes = np.unique(y_train)
    cls_w = compute_class_weight('balanced', classes=classes, y=y_train)
    sample_weight = cls_w[np.searchsorted(classes, y_train)]

    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )
    # NOTE (Colab sklearn < 1.4): MLPClassifier.fit() gained the
    # sample_weight argument only in scikit-learn 1.4. Colab's image
    # (older sklearn) crashes with "unexpected keyword argument
    # 'sample_weight'", so detect support at runtime instead. When it is
    # unsupported we fall back to unweighted training and say so, so the
    # imbalance handling of the MLP baseline is never silently dropped.
    if 'sample_weight' in inspect.signature(model.fit).parameters:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        print("[mlp] sklearn does not support sample_weight; training "
              "without per-sample balancing (see src/baselines/mlp.py)")
        model.fit(X_train, y_train)

    metrics = evaluate_tabular(model, X_test, y_test, num_classes)
    metrics['model'] = 'mlp'
    return metrics
