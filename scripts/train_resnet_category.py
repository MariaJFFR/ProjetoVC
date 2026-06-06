import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(ROOT / "src"))

from category_protocol import (
    category_train_good_paths,
    category_test_records,
)

from bottle_protocol import (
    split_train_validation_good,
)

from eval_resnet import (
    build_resnet18_feature_extractor,
    extract_features,
    fit_mahalanobis,
    mahalanobis_scores,
)

from metrics import (
    binary_classification_metrics,
    safe_auroc,
)

from utils import (
    get_device,
    save_json,
    set_seed,
)


CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "tile",
    "wood",
]


def parse_args():

    parser = argparse.ArgumentParser(
        description="Train and evaluate ResNet18 + Mahalanobis for multiple MVTec categories."
    )

    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--threshold-quantile",
        type=float,
        default=0.95,
    )

    return parser.parse_args()


def evaluate_category(
    category,
    device,
    val_fraction,
    seed,
    threshold_quantile,
):

    print(
        f"\n{'=' * 60}"
    )

    print(
        f"Category: {category}"
    )

    train_good = category_train_good_paths(
        category
    )

    train_paths, val_paths = split_train_validation_good(
        train_good,
        val_fraction=val_fraction,
        seed=seed,
    )

    records = category_test_records(
        category
    )

    test_paths = [
        r.path
        for r in records
    ]

    test_labels = np.asarray(
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

    train_features = extract_features(
        train_paths,
        resnet,
        preprocess,
        device,
    )

    val_features = extract_features(
        val_paths,
        resnet,
        preprocess,
        device,
    )

    test_features = extract_features(
        test_paths,
        resnet,
        preprocess,
        device,
    )

    estimator = fit_mahalanobis(
        train_features
    )

    val_scores = mahalanobis_scores(
        estimator,
        val_features,
    )

    threshold = float(
        np.quantile(
            val_scores,
            threshold_quantile,
        )
    )

    test_scores = mahalanobis_scores(
        estimator,
        test_features,
    )

    metrics = binary_classification_metrics(
        test_labels,
        test_scores,
        threshold,
    )

    metrics["image_auroc"] = safe_auroc(
        test_labels,
        test_scores,
    )

    model_path = (
        ROOT
        / "models"
        / f"resnet_mahalanobis_{category}.joblib"
    )

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "category": category,
            "estimator": estimator,
            "threshold": threshold,
            "threshold_quantile": threshold_quantile,
            "seed": seed,
        },
        model_path,
    )

    output = {
        "category": category,
        "model": "resnet18_mahalanobis",
        "seed": seed,
        "threshold_quantile": threshold_quantile,
        "threshold": threshold,
        "metrics": metrics,
        "model_path": str(model_path),
    }

    out_path = (
        ROOT
        / "results"
        / f"resnet_{category}_metrics.json"
    )

    save_json(
        output,
        out_path,
    )

    print(
        f"AUROC: {metrics['image_auroc']:.4f}"
    )

    print(
        f"F1: {metrics['f1']:.4f}"
    )

    return {
        "category": category,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "image_auroc": metrics["image_auroc"],
    }


def main():

    args = parse_args()

    set_seed(
        args.seed
    )

    device = get_device()

    summary = []

    for category in CATEGORIES:

        result = evaluate_category(
            category=category,
            device=device,
            val_fraction=args.val_fraction,
            seed=args.seed,
            threshold_quantile=args.threshold_quantile,
        )

        summary.append(
            result
        )

    df = pd.DataFrame(
        summary
    )

    summary_path = (
        ROOT
        / "results"
        / "resnet_summary.csv"
    )

    df.to_csv(
        summary_path,
        index=False,
    )

    print(
        "\nSummary:"
    )

    print(df)

    print(
        f"\nSaved summary: {summary_path}"
    )


if __name__ == "__main__":
    main()