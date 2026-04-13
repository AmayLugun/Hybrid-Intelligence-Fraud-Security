import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score

def _setup_ax(figsize=(8, 5), title=None):
    """Helper to create a figure and axis."""
    fig, ax = plt.subplots(figsize=figsize)
    if title: ax.set_title(title)
    return fig, ax

def plot_heatmap(df):
    fig, ax = _setup_ax(figsize=(10, 8), title="Correlation Heatmap")
    sns.heatmap(df.corr(), cmap='coolwarm', ax=ax)
    return fig

def plot_precision_recall_curve(precisions, recalls, average_precision=None, ax=None):
    if ax is None:
        fig, ax = _setup_ax(figsize=(6, 4))
    else:
        fig = ax.get_figure()

    label = f"PR Curve (AP = {average_precision:.4f})" if average_precision else "PR Curve"
    ax.step(recalls, precisions, where='post', label=label, linewidth=2)
    ax.fill_between(recalls, precisions, step='post', alpha=0.2)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(ax.get_title() or 'Precision-Recall Curve')
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    return fig

def plot_feature_importance(importances, feature_names):
    idx = np.argsort(importances)[::-1]
    fig, ax = _setup_ax(figsize=(8, 4), title="Feature Importance")
    ax.barh(range(len(idx)), importances[idx])
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels(feature_names[idx])
    ax.invert_yaxis()
    return fig

def plot_confusion_matrix(cm, classes=['Normal', 'Fraud'], title='Confusion Matrix', cmap='Blues', ax=None):
    if ax is None: fig, ax = _setup_ax(figsize=(5, 4))
    else: fig = ax.get_figure()
    
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title(title)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    return fig

def plot_model_performance(y_true, y_prob, labels=['Normal', 'Fraud'], title_suffix='Model'):
    y_pred = (y_prob > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    plot_confusion_matrix(cm, classes=labels, title=f'{title_suffix} Confusion Matrix', ax=ax1)
    plot_precision_recall_curve(prec, rec, average_precision=ap, ax=ax2)
    ax2.set_title(f'{title_suffix} PR Curve')
    
    plt.tight_layout()
    return fig

def plot_top_correlations(df, target_col='Class', top_n=10):
    correlations = df.corr()[target_col].drop(target_col)
    top_corr = correlations.abs().sort_values(ascending=False).head(top_n)
    top_vals = correlations[top_corr.index]
    
    fig, ax = _setup_ax(figsize=(8, 4), title=f'Top {top_n} Correlations with {target_col}')
    colors = ['#FF4B4B' if x < 0 else '#2196F3' for x in top_vals]
    top_vals.plot(kind='barh', ax=ax, color=colors)
    ax.set_xlabel('Correlation Coefficient')
    ax.invert_yaxis()
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    return fig

def plot_class_distribution(df, feature, target_col='Class'):
    fig, ax = _setup_ax(figsize=(7, 3.5), title=f'Distribution of {feature}: Fraud vs Legitimate')
    for label, color in [ (0, '#2196F3'), (1, '#FF4B4B') ]:
        subset = df[df[target_col] == label][feature]
        sns.kdeplot(subset, label='Fraud' if label else 'Legitimate', fill=True, color=color, ax=ax)
    ax.set_xlabel(feature)
    ax.legend()
    return fig