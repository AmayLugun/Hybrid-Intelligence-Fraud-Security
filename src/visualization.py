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
"""
def plot_precision_recall_curve(precision, recall, avg_precision):
    fig, ax = plt.subplots()
    ax.plot(recall, precision, label=f'Avg Precision: {avg_precision:.2f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend()
    return fig 
"""