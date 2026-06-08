"""Task 2 - Defect type identification.

For every MVTec category this trains a multiclass classifier that predicts the
*defect type* (e.g. broken_large, broken_small, contamination) of a defective
image. It reuses the ImageNet-pretrained ResNet18 feature extractor already used
by the anomaly-detection stack and fits a LogisticRegression head on top.

MVTec AD does not provide defective training images: defects only exist under
`test/`. The defective test images are therefore split into a train and a test
subset (stratified by defect type). Because of that the reported numbers are an
illustrative defect-type baseline and are not part of the official MVTec
anomaly-detection protocol used by the unsupervised methods.
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(ROOT / "src"))

from category_protocol import (
    category_defective_records,
    category_defect_types,
)

from eval_resnet import (
    build_resnet18_feature_extractor,
    extract_features,
)

from metrics import (
    multiclass_metrics,
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
        description="Train and evaluate a ResNet18 + LogisticRegression defect-type classifier per MVTec category."
    )

    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.3,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def evaluate_category(
    category,
    device,
    resnet,
    preprocess,
    test_fraction,
    seed,
):

    print(
        f"\n{'=' * 60}"
    )

    print(
        f"Category: {category}"
    )

    defect_types = category_defect_types(
        category
    )

    if len(defect_types) < 2:

        print(
            f"Skipped: only {len(defect_types)} defect type(s) "
            f"({defect_types}); multiclass classification is degenerate."
        )

        return None

    records = category_defective_records(
        category
    )

    paths = [
        record.path
        for record in records
    ]

    targets = np.asarray(
        [
            record.defect_type
            for record in records
        ]
    )

    train_paths, test_paths, train_targets, test_targets = train_test_split(
        paths,
        targets,
        test_size=test_fraction,
        random_state=seed,
        stratify=targets,
    )

    train_features = extract_features(
        train_paths,
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

    classifier = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=seed,
    )

    classifier.fit(
        train_features,
        train_targets,
    )

    test_predictions = classifier.predict(
        test_features
    )

    metrics = multiclass_metrics(
        test_targets,
        test_predictions,
        labels=defect_types,
    )

    model_path = (
        ROOT
        / "models"
        / f"defect_classifier_{category}.joblib"
    )

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "category": category,
            "classifier": classifier,
            "defect_types": defect_types,
            "seed": seed,
        },
        model_path,
    )

    output = {
        "category": category,
        "model": "resnet18_logreg_defect_type",
        "seed": seed,
        "test_fraction": test_fraction,
        "defect_types": defect_types,
        "n_train": len(train_paths),
        "n_test": len(test_paths),
        "metrics": metrics,
        "model_path": str(model_path),
    }

    out_path = (
        ROOT
        / "results"
        / f"defect_classifier_{category}_metrics.json"
    )

    save_json(
        output,
        out_path,
    )

    print(
        f"Defect types: {defect_types}"
    )

    print(
        f"Accuracy: {metrics['accuracy']:.4f}"
    )

    print(
        f"Macro F1: {metrics['macro_f1']:.4f}"
    )

    return {
        "category": category,
        "n_classes": len(defect_types),
        "n_train": len(train_paths),
        "n_test": len(test_paths),
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
    }


def main():

    args = parse_args()

    set_seed(
        args.seed
    )

    device = get_device()

    resnet, preprocess = build_resnet18_feature_extractor(
        device
    )

    summary = []

    for category in CATEGORIES:

        result = evaluate_category(
            category=category,
            device=device,
            resnet=resnet,
            preprocess=preprocess,
            test_fraction=args.test_fraction,
            seed=args.seed,
        )

        if result is not None:

            summary.append(
                result
            )

    if not summary:

        print(
            "\nNo category produced a defect-type classifier."
        )

        return

    df = pd.DataFrame(
        summary
    )

    summary_path = (
        ROOT
        / "results"
        / "defect_classifier_summary.csv"
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
