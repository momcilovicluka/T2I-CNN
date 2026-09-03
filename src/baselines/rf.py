"""Random Forest baseline for tabular data classification."""

from sklearn.ensemble import RandomForestClassifier
from src.evaluate import evaluate_tabular


def train_and_evaluate(X_train, y_train, X_test, y_test, num_classes=2):
    """Train Random Forest and return metrics dict.

    WHY RF as baseline:
    - Industry standard for tabular data
    - Handles mixed feature types natively
    - Provides feature importance rankings
    - No hyperparameter tuning needed for reasonable defaults
    - class_weight='balanced' mirrors the inverse-frequency class
      weights used in the CNN loss (professor-validation 10.1): without
      it, baselines on imbalanced sets (adult ~3:1, dry bean ~6.8:1)
      would be compared to CNNs that were given an imbalance remedy the
      baselines were denied.
    """
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,        # Let trees grow fully
        min_samples_split=2,
        class_weight='balanced',  # parity with CNN class weights
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    metrics = evaluate_tabular(model, X_test, y_test, num_classes)
    metrics['model'] = 'random_forest'
    return metrics
