import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(ROOT / "src"))

from category_protocol import (
    category_test_records,
)

from eval_resnet import (
    build_resnet18_feature_extractor,
    extract_features,
    mahalanobis_scores,
)

from utils import (
    get_device,
)

CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "tile",
    "wood",
]


def best_f1_threshold(labels, scores):

    thresholds = np.unique(scores)

    best_f1 = -1
    best_threshold = None

    for t in thresholds:

        preds = (scores > t).astype(int)

        f1 = f1_score(
            labels,
            preds,
            zero_division=0,
        )

        if f1 > best_f1:

            best_f1 = f1
            best_threshold = float(t)

    return best_threshold, best_f1


def best_youden_threshold(labels, scores):

    fpr, tpr, thresholds = roc_curve(
        labels,
        scores,
    )

    idx = np.argmax(
        tpr - fpr
    )

    return float(thresholds[idx])


def analyze_category(
    category,
    device,
):

    print(
        f"\n{'='*70}"
    )

    print(
        f"CATEGORY: {category.upper()}"
    )

    detector_path = (
        ROOT /
        "models" /
        f"resnet_mahalanobis_{category}.joblib"
    )

    detector = joblib.load(
        detector_path
    )

    current_threshold = detector[
        "threshold"
    ]

    records = category_test_records(
        category
    )

    paths = [
        r.path
        for r in records
    ]

    labels = np.asarray(
        [
            r.label
            for r in records
        ]
    )

    resnet, preprocess = (
        build_resnet18_feature_extractor(
            device
        )
    )

    features = extract_features(
        paths,
        resnet,
        preprocess,
        device,
    )

    scores = mahalanobis_scores(
        detector["estimator"],
        features,
    )

    normal_scores = scores[
        labels == 0
    ]

    defect_scores = scores[
        labels == 1
    ]

    auroc = roc_auc_score(
        labels,
        scores,
    )

    current_preds = (
        scores > current_threshold
    ).astype(int)

    current_f1 = f1_score(
        labels,
        current_preds,
        zero_division=0,
    )

    best_f1_thr, best_f1 = (
        best_f1_threshold(
            labels,
            scores,
        )
    )

    best_youden_thr = (
        best_youden_threshold(
            labels,
            scores,
        )
    )

    print(
        f"Current Threshold : {current_threshold:.2f}"
    )

    print(
        f"Mean Normal Score : {normal_scores.mean():.2f}"
    )

    print(
        f"Mean Defect Score : {defect_scores.mean():.2f}"
    )

    print(
        f"AUROC             : {auroc:.4f}"
    )

    print(
        f"Current F1        : {current_f1:.4f}"
    )

    print(
        f"Best F1 Threshold : {best_f1_thr:.2f}"
    )

    print(
        f"Best F1           : {best_f1:.4f}"
    )

    print(
        f"Best Youden Thr   : {best_youden_thr:.2f}"
    )

    return {
        "category": category,
        "current_threshold": current_threshold,
        "best_f1_threshold": best_f1_thr,
        "best_youden_threshold": best_youden_thr,
        "current_f1": current_f1,
        "best_f1": best_f1,
        "auroc": auroc,
        "normal_mean": normal_scores.mean(),
        "defect_mean": defect_scores.mean(),
    }


def main():

    device = get_device()

    rows = []

    for category in CATEGORIES:

        row = analyze_category(
            category,
            device,
        )

        rows.append(
            row
        )

    df = pd.DataFrame(
        rows
    )

    print(
        "\n\nSUMMARY\n"
    )

    print(df)

    output = (
        ROOT /
        "results" /
        "threshold_analysis.csv"
    )

    df.to_csv(
        output,
        index=False,
    )

    print(
        f"\nSaved: {output}"
    )


if __name__ == "__main__":
    main()