"""XGBoost baseline for tabular data classification."""

from xgboost import XGBClassifier
from src.evaluate import evaluate_tabular


def train_and_evaluate(X_train, y_train, X_test, y_test, num_classes=2):
    """Train XGBoost and return metrics dict.

    WHY XGBoost as baseline:
    - State-of-the-art for tabular data (consistently wins Kaggle)
    - Gradient boosting captures non-linear interactions
    - Often outperforms deep learning on structured data
    - If CNN+T2I can't beat XGBoost, the transformation is lossy
    """
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss',
        verbosity=0,
    )
    model.fit(X_train, y_train)

    metrics = evaluate_tabular(model, X_test, y_test, num_classes)
    metrics['model'] = 'xgboost'
    return metrics
