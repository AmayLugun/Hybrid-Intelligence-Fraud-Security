# Comprehensive model evaluation utilities
from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)


def evaluate_model(y_true, y_pred, y_prob=None):

    results = {
        "recall": recall_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }

    if y_prob is not None:
        results["auc_roc"] = roc_auc_score(y_true, y_prob)
        results["avg_precision"] = average_precision_score(y_true, y_prob)

    return results


def get_roc_curve(y_true, y_prob):
    """Return (fpr, tpr, thresholds) for an ROC curve."""
    return roc_curve(y_true, y_prob)


def get_pr_curve(y_true, y_prob):
    """Return (precision, recall, thresholds) for a Precision-Recall curve."""
    return precision_recall_curve(y_true, y_prob)
