"""XGBoost baseline for tabular data classification."""

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
from src.evaluate import evaluate_tabular


def train_and_evaluate(X_train, y_train, X_test, y_test, num_classes=2):
    """Train XGBoost and return metrics dict.

    WHY XGBoost as baseline:
    - State-of-the-art for tabular data (consistently wins Kaggle)
    - Gradient boosting captures non-linear interactions
    - Often outperforms deep learning on structured data
    - If CNN+T2I can't beat XGBoost, the transformation is lossy
    - Per-sample balanced weights mirror the inverse-frequency class
      weights used in the CNN loss (professor-validation 10.1), so the
      baseline is not handicapped on imbalanced sets.
    """
    # Balanced per-sample weights (same remedy the CNN loss receives).
    classes = np.unique(y_train)
    cls_w = compute_class_weight('balanced', classes=classes, y=y_train)
    sample_weight = cls_w[np.searchsorted(classes, y_train)]

    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss',
        verbosity=0,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)

    metrics = evaluate_tabular(model, X_test, y_test, num_classes)
    metrics['model'] = 'xgboost'
    return metrics
