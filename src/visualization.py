import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_heatmap(df):
    fig, ax = plt.subplots(figsize=(12,10))
    sns.heatmap(df.corr(), cmap='coolwarm',ax=ax)
    ax.set_title("Correlation Heatmap")
    return fig

def plot_precision_recall_curve(precisions, recalls, average_precision=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    label = "Precision-Recall Curve"
    if average_precision is not None:
        label += f" (AP = {average_precision:.4f})"
        
    ax.step(recalls, precisions, where='post', label=label, linewidth=2)
    ax.fill_between(recalls, precisions, step='post', alpha=0.2)
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    
    return fig

def plot_feature_importance(importances, feature_names):
    idx = np.argsort(importances)[::-1]
    
    fig, ax = plt.subplots()
    ax.barh(range(len(idx)), importances[idx])
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels(feature_names[idx])
    ax.set_title("Feature Importance")
    
    return fig

def plot_confusion_matrix(cm, classes=['Normal', 'Fraud'], title='Confusion Matrix', cmap='Blues', ax=None):
    if ax is None: fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title(title)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    return ax.figure

def plot_model_performance(y_true, y_prob, labels=['Normal', 'Fraud'], title_suffix='Model'):
    from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
    
    y_pred = (y_prob > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax1)
    ax1.set_title(f'{title_suffix} Confusion Matrix')
    ax1.set_ylabel('True label')
    ax1.set_xlabel('Predicted label')
    
    # 2. PR Curve
    ax2.step(rec, prec, where='post', label=f'AP = {ap:.4f}')
    ax2.fill_between(rec, prec, alpha=0.2)
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title(f'{title_suffix} PR Curve')
    ax2.legend()
    
    plt.tight_layout()
    return fig