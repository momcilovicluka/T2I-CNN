"""
Model evaluation with comprehensive metrics.

Primary metric: macro-F1 (handles class imbalance).
Also reports: accuracy, precision, recall, ROC-AUC, PR-AUC, confusion matrix.
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)


def evaluate_model(model, test_loader, num_classes, device='cpu'):
    """
    Evaluate a trained CNN model on test data.

    Returns dict:
        accuracy, precision_macro, recall_macro, f1_macro,
        roc_auc, pr_auc, confusion_matrix, classification_report
    """
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            output = model(X_batch)
            probs = torch.softmax(output, dim=1)
            _, predicted = output.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    return _compute_metrics(all_labels, all_preds, all_probs, num_classes)


def evaluate_tabular(model, X_test, y_test, num_classes):
    """Evaluate a sklearn/xgboost model on test data."""
    preds = model.predict(X_test)

    # Get probabilities if available
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_test)
    else:
        probs = None

    return _compute_metrics(y_test, preds, probs, num_classes)


def _compute_metrics(y_true, y_pred, y_probs, num_classes):
    """Compute all metrics from true labels, predictions, and probabilities."""
    results = {}

    # Accuracy
    results['accuracy'] = accuracy_score(y_true, y_pred)

    # Macro-averaged precision, recall, F1 (handles imbalance)
    avg = 'macro' if num_classes > 2 else 'binary'
    results['precision_macro'] = precision_score(y_true, y_pred, average=avg, zero_division=0)
    results['recall_macro'] = recall_score(y_true, y_pred, average=avg, zero_division=0)
    results['f1_macro'] = f1_score(y_true, y_pred, average=avg, zero_division=0)

    # ROC-AUC and PR-AUC (need probabilities)
    if y_probs is not None and y_probs.ndim == 2:
        try:
            if num_classes == 2:
                results['roc_auc'] = roc_auc_score(y_true, y_probs[:, 1])
                results['pr_auc'] = average_precision_score(y_true, y_probs[:, 1])
            else:
                results['roc_auc'] = roc_auc_score(
                    y_true, y_probs, multi_class='ovr', average='macro'
                )
                results['pr_auc'] = average_precision_score(
                    y_true, y_probs, average='macro'
                )
        except Exception:
            results['roc_auc'] = 0.0
            results['pr_auc'] = 0.0
    else:
        results['roc_auc'] = 0.0
        results['pr_auc'] = 0.0

    # Confusion matrix
    results['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
    results['classification_report'] = classification_report(
        y_true, y_pred, zero_division=0
    )

    return results
