import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import seaborn as sns
import os

from sklearn.metrics import roc_curve, precision_recall_curve

def plot_and_save_metrics(metrics: dict, model_name='model', save_dir='xgboost/imgs'):
    os.makedirs(save_dir, exist_ok=True)
    prefix = model_name.replace(" ", "_").lower()

    y_true = metrics['y_true']
    y_proba = metrics['y_proba']

    # === Confusion Matrix ===
    cm = metrics['confusion_matrix_best'] if 'confusion_matrix_best' in metrics else metrics['confusion_matrix']
    cm_title = f"{model_name} - Confusion Matrix"
    cm_filename = f"{prefix}_confusion_matrix.png"

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(cm_title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, cm_filename))
    plt.close()

    # === ROC Curve ===
    if metrics.get('roc_auc') is not None:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC AUC = {metrics['roc_auc']:.2f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{model_name} - ROC Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{prefix}_roc_curve.png"))
        plt.close()

    # === Precision-Recall Curve ===
    if metrics.get('pr_auc') is not None:
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        plt.figure()
        plt.plot(recall, precision, label=f"PR AUC = {metrics['pr_auc']:.2f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"{model_name} - Precision-Recall Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{prefix}_pr_curve.png"))
        plt.close()

print("✅ plot_save_imgs.py ran successfully!")
