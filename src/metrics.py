import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def safe_auroc(y_true, scores):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, scores))


def binary_classification_metrics(y_true, scores, threshold):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores)
    y_pred = (scores >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def multiclass_metrics(y_true, y_pred, labels):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "per_class": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=labels,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=labels,
        ).tolist(),
        "labels": list(labels),
    }


def segmentation_metrics(pixel_labels, pixel_scores, threshold):
    pixel_labels = np.asarray(pixel_labels).astype(np.uint8)
    pixel_scores = np.asarray(pixel_scores)
    pred = (pixel_scores >= threshold).astype(np.uint8)
    inter = int((pred & pixel_labels).sum())
    union = int((pred | pixel_labels).sum())
    pred_sum = int(pred.sum())
    gt_sum = int(pixel_labels.sum())
    dice_den = pred_sum + gt_sum
    return {
        "pixel_threshold": float(threshold),
        "pixel_precision": float(precision_score(pixel_labels, pred, zero_division=0)),
        "pixel_recall": float(recall_score(pixel_labels, pred, zero_division=0)),
        "pixel_f1": float(f1_score(pixel_labels, pred, zero_division=0)),
        "iou": float(inter / union) if union else 0.0,
        "dice": float(2 * inter / dice_den) if dice_den else 0.0,
    }
