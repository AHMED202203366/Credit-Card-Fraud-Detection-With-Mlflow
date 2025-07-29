import numpy as np
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)

def evaluate_model_default_threshold(model, X, y, threshold=0.5):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    y_pred_default = (y_prob >= threshold).astype(int)

    metrics = {
        "threshold_default": threshold,
        "y_pred_default": y_pred_default,
        "accuracy": accuracy_score(y, y_pred_default),
        "precision": precision_score(y, y_pred_default, zero_division=0),
        "recall": recall_score(y, y_pred_default, zero_division=0),
        "f1_score": f1_score(y, y_pred_default, zero_division=0),
        "roc_auc": roc_auc_score(y, y_prob) if np.any(y_prob) else None,
        "pr_auc": average_precision_score(y, y_prob) if np.any(y_prob) else None,
        "confusion_matrix": confusion_matrix(y, y_pred_default),
        "classification_report": classification_report(y, y_pred_default, output_dict=True),
        "y_pred": y_pred,
        "y_proba": y_prob,
        "y_true": y
    }

    return metrics

def find_best_threshold(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    precision, recall = precision[:-1], recall[:-1]
    f1_scores = (2 * precision * recall) / (precision + recall + 1e-8)

    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_precision = precision[best_idx]
    best_recall = recall[best_idx]
    best_f1 = f1_scores[best_idx]

    return best_threshold, best_precision, best_recall, best_f1

def evaluate_model_custom_threshold(model, X, y, use_fixed_threshold=None, return_only_threshold=False):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    if use_fixed_threshold is not None:
        best_threshold = float(use_fixed_threshold)
        y_pred_best = (y_prob >= best_threshold).astype(int)
        best_precision = precision_score(y, y_pred_best, zero_division=0)
        best_recall = recall_score(y, y_pred_best, zero_division=0)
        best_f1 = f1_score(y, y_pred_best, zero_division=0)
    else:
        best_threshold, best_precision, best_recall, best_f1 = find_best_threshold(y, y_prob)
        y_pred_best = (y_prob >= best_threshold).astype(int)

    if return_only_threshold:
        return {}, best_threshold

    metrics = {
        "best_threshold": best_threshold,
        "precision_best": best_precision,
        "recall_best": best_recall,
        "best_f1_score": best_f1,
        "roc_auc": roc_auc_score(y, y_prob) if np.any(y_prob) else None,
        "pr_auc": average_precision_score(y, y_prob) if np.any(y_prob) else None,
        "confusion_matrix_best": confusion_matrix(y, y_pred_best),
        "classification_report_best": classification_report(y, y_pred_best, output_dict=True),
        "y_pred": y_pred,
        "y_proba": y_prob,
        "y_true": y
    }

    return metrics, best_threshold

def print_evaluation_metrics(
    metrics: dict,
    title: str,
    model_name: str,
    save_to_txt: bool = False,
    txt_filename: str = "classification_report.txt",
    save_dir: str = "xgboost/reports"
):
    report_lines = []
    report_lines.append(f"\n========== Evaluation Report {title} =============")

    if "threshold_default" in metrics:
        report_lines.append("\n--- Default Threshold Evaluation ---")
        report_lines.append(f"Threshold       : {metrics['threshold_default']}")
        report_lines.append(f"Accuracy        : {metrics['accuracy']:.4f}")
        report_lines.append(f"Precision       : {metrics['precision']:.4f}")
        report_lines.append(f"Recall          : {metrics['recall']:.4f}")
        report_lines.append(f"F1 Score        : {metrics['f1_score']:.4f}")
        report_lines.append(f"ROC AUC         : {metrics['roc_auc']:.4f}" if metrics["roc_auc"] is not None else "ROC AUC: N/A")
        report_lines.append(f"PR AUC          : {metrics['pr_auc']:.4f}" if metrics["pr_auc"] is not None else "PR AUC: N/A")
        report_lines.append(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
        report_lines.append(f"Classification Report:\n{classification_report(metrics['y_true'], metrics['y_pred_default'])}")

    if "best_threshold" in metrics:
        report_lines.append("\n--- Best Threshold Evaluation ---")
        report_lines.append(f"Best Threshold  : {metrics['best_threshold']:.4f}")
        report_lines.append(f"Best Precision  : {metrics['precision_best']:.4f}")
        report_lines.append(f"Best Recall     : {metrics['recall_best']:.4f}")
        report_lines.append(f"Best F1 Score   : {metrics['best_f1_score']:.4f}")
        report_lines.append(f"ROC AUC         : {metrics['roc_auc']:.4f}" if metrics["roc_auc"] is not None else "ROC AUC: N/A")
        report_lines.append(f"PR AUC          : {metrics['pr_auc']:.4f}" if metrics["pr_auc"] is not None else "PR AUC: N/A")
        report_lines.append(f"Confusion Matrix:\n{metrics['confusion_matrix_best']}")
        report_lines.append(f"Classification Report:\n{classification_report(metrics['y_true'], (metrics['y_proba'] >= metrics['best_threshold']).astype(int))}")

    # Print to console
    print("\n".join(report_lines))

    if save_to_txt:
        os.makedirs(save_dir, exist_ok=True)
        prefix = model_name.replace(" ", "_").lower()
        txt_path = os.path.join(save_dir, f"{prefix}_{txt_filename}")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
        print(f"\n✅ Saved evaluation report to: {txt_path}")

print("✅ evalute_metrics.py ran successfully!")