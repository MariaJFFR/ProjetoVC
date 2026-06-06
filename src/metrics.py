import numpy as np
from sklearn.metrics import (
    accuracy_score,
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
