# Backward-compatible wrapper — use src.evaluation for new code
#
# The old evaluate_model(y_true, y_pred) returned (recall, cm).
# The new one returns a dict. This wrapper preserves the old API.

from sklearn.metrics import recall_score, confusion_matrix


def evaluate_model(y_true, y_pred):
    """Legacy evaluate_model — returns (recall, confusion_matrix)."""
    recall = recall_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    return recall, cm