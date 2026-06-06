import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from autoencoder import Autoencoder
from bottle_protocol import (
    bottle_test_records,
    bottle_train_good_paths,
    split_train_validation_good,
)
from dataset import criar_loader_de_paths
from eval_autoencoder import (
    flatten_test_masks,
    reconstruction_maps,
    validation_pixel_errors,
)
from metrics import binary_classification_metrics, safe_auroc, segmentation_metrics
from utils import get_device, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Bottle autoencoder.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-threshold-quantile", type=float, default=0.95)
    parser.add_argument("--pixel-threshold-quantile", type=float, default=0.995)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    model_path = ROOT / "models" / "autoencoder_bottle_best.pth"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing {model_path}. Run scripts/train_autoencoder_bottle.py first."
        )

    model = Autoencoder().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    train_good = bottle_train_good_paths()
    _, val_paths = split_train_validation_good(
        train_good,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    val_loader = criar_loader_de_paths(
        val_paths,
        batch_size=args.batch_size,
        tamanho_imagem=args.image_size,
        shuffle=False,
    )
    val_outputs = reconstruction_maps(model, val_loader, device)
    image_threshold = float(
        np.quantile(val_outputs["image_scores"], args.image_threshold_quantile)
    )
    val_pixel_scores = validation_pixel_errors(
        model,
        val_paths,
        device=device,
        image_size=args.image_size,
    )
    pixel_threshold = float(np.quantile(val_pixel_scores, args.pixel_threshold_quantile))

    records = bottle_test_records()
    test_paths = [r.path for r in records]
    test_labels = [r.label for r in records]
    test_types = [r.defect_type for r in records]
    test_loader = criar_loader_de_paths(
        test_paths,
        labels=test_labels,
        tipos=test_types,
        batch_size=args.batch_size,
        tamanho_imagem=args.image_size,
        shuffle=False,
    )
    test_outputs = reconstruction_maps(model, test_loader, device)

    pixel_scores = np.concatenate([m.ravel() for m in test_outputs["error_maps"]])
    pixel_labels = flatten_test_masks(
        test_outputs["paths"],
        size=args.image_size,
    )

    image_metrics = binary_classification_metrics(
        test_outputs["labels"],
        test_outputs["image_scores"],
        image_threshold,
    )
    image_metrics["image_auroc"] = safe_auroc(
        test_outputs["labels"],
        test_outputs["image_scores"],
    )
    pixel_metrics = segmentation_metrics(pixel_labels, pixel_scores, pixel_threshold)
    pixel_metrics["pixel_auroc"] = safe_auroc(pixel_labels, pixel_scores)

    per_type = {}
    for defect_type in ["good", "broken_large", "broken_small", "contamination"]:
        mask = test_outputs["defect_types"] == defect_type
        n = int(mask.sum())
        predicted_defect = int((test_outputs["image_scores"][mask] >= image_threshold).sum())
        per_type[defect_type] = {
            "images": n,
            "predicted_defect": predicted_defect,
        }

    output = {
        "category": "bottle",
        "model": "autoencoder",
        "protocol": "train on train/good split; thresholds calibrated on train/good validation; final evaluation on test only",
        "seed": args.seed,
        "image_size": args.image_size,
        "image_threshold_quantile": args.image_threshold_quantile,
        "pixel_threshold_quantile": args.pixel_threshold_quantile,
        "image_threshold": image_threshold,
        "pixel_threshold": pixel_threshold,
        "image_metrics": image_metrics,
        "pixel_metrics": pixel_metrics,
        "per_defect_type_counts": per_type,
    }
    out_path = ROOT / "results" / "autoencoder_bottle_metrics.json"
    save_json(output, out_path)
    print(f"Saved metrics: {out_path}")
    print(output)


if __name__ == "__main__":
    main()
