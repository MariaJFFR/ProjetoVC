import argparse
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from bottle_protocol import (
    bottle_test_records,
    bottle_train_good_paths,
    split_train_validation_good,
)
from eval_resnet import (
    build_resnet18_feature_extractor,
    extract_features,
    fit_mahalanobis,
    mahalanobis_scores,
)
from metrics import binary_classification_metrics, safe_auroc
from utils import get_device, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Bottle ResNet18 + Mahalanobis.")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold-quantile", type=float, default=0.95)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    train_good = bottle_train_good_paths()
    train_paths, val_paths = split_train_validation_good(
        train_good,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    records = bottle_test_records()
    test_paths = [r.path for r in records]
    test_labels = np.asarray([r.label for r in records])

    resnet, preprocess = build_resnet18_feature_extractor(device)
    train_features = extract_features(train_paths, resnet, preprocess, device)
    val_features = extract_features(val_paths, resnet, preprocess, device)
    test_features = extract_features(test_paths, resnet, preprocess, device)

    estimator = fit_mahalanobis(train_features)
    val_scores = mahalanobis_scores(estimator, val_features)
    threshold = float(np.quantile(val_scores, args.threshold_quantile))
    test_scores = mahalanobis_scores(estimator, test_features)

    metrics = binary_classification_metrics(test_labels, test_scores, threshold)
    metrics["image_auroc"] = safe_auroc(test_labels, test_scores)

    model_path = ROOT / "models" / "resnet_mahalanobis_bottle.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "estimator": estimator,
            "threshold": threshold,
            "threshold_quantile": args.threshold_quantile,
            "seed": args.seed,
        },
        model_path,
    )

    output = {
        "category": "bottle",
        "model": "resnet18_mahalanobis",
        "protocol": "fit on train/good split; threshold calibrated on train/good validation; final evaluation on test only",
        "seed": args.seed,
        "threshold_quantile": args.threshold_quantile,
        "threshold": threshold,
        "metrics": metrics,
        "model_path": str(model_path),
    }
    out_path = ROOT / "results" / "resnet_bottle_metrics.json"
    save_json(output, out_path)
    print(f"Saved estimator: {model_path}")
    print(f"Saved metrics: {out_path}")
    print(output)


if __name__ == "__main__":
    main()
