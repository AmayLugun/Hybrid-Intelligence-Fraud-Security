# evaluation_legacy.py — Backward-compatible wrapper
# Use src.evaluation for new code (returns dict).
# This module preserves the old API (returns tuple).

from sklearn.metrics import recall_score, confusion_matrix


def evaluate_model(y_true, y_pred):
    """Legacy evaluate_model — returns (recall, confusion_matrix)."""
    recall = recall_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    return recall, cm