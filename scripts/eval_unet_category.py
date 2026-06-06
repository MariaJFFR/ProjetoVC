import sys

from pathlib import Path

import numpy as np
import pandas as pd

import torch

ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(ROOT / "src"))

from unet import UNet

from metrics import (
    segmentation_metrics,
    safe_auroc,
)

from utils import (
    get_device,
    save_json,
)

CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "tile",
    "wood",
]


def evaluate_category(
    category,
    device,
):

    print(
        f"\n{'='*60}"
    )

    print(
        f"Evaluating U-Net: {category}"
    )

    test_ds = torch.load(
        ROOT
        / "models"
        / f"unet_{category}_test_split.pt",
        weights_only=False,
    )

    model = UNet().to(
        device
    )

    model.load_state_dict(
        torch.load(
            ROOT
            / "models"
            / f"unet_{category}_best.pth",
            map_location=device,
        )
    )

    model.eval()

    pixel_scores = []
    pixel_labels = []

    with torch.no_grad():

        for image, mask in test_ds:

            image = (
                image
                .unsqueeze(0)
                .to(device)
            )

            pred = torch.sigmoid(
                model(image)
            )

            pixel_scores.extend(
                pred
                .cpu()
                .numpy()
                .ravel()
            )

            pixel_labels.extend(
                mask
                .numpy()
                .ravel()
            )

    pixel_scores = np.asarray(
        pixel_scores
    )

    pixel_labels = np.asarray(
        pixel_labels
    )
    
    print(
        "min:",
        pixel_scores.min()
    )

    print(
        "max:",
        pixel_scores.max()
    )

    print(
        "mean:",
        pixel_scores.mean()
    )

    threshold = 0.5

    metrics = segmentation_metrics(
        pixel_labels,
        pixel_scores,
        threshold,
    )

    metrics["pixel_auroc"] = (
        safe_auroc(
            pixel_labels,
            pixel_scores,
        )
    )

    output = {
        "category": category,
        "model": "unet",
        "threshold": threshold,
        "metrics": metrics,
    }

    save_json(
        output,
        ROOT
        / "results"
        / f"unet_{category}_metrics.json",
    )

    print(
        f"Pixel AUROC: {metrics['pixel_auroc']:.4f}"
    )

    print(
        f"IoU: {metrics['iou']:.4f}"
    )

    print(
        f"Dice: {metrics['dice']:.4f}"
    )

    return {
        "category": category,
        "pixel_auroc": metrics["pixel_auroc"],
        "iou": metrics["iou"],
        "dice": metrics["dice"],
    }


def main():

    device = get_device()

    summary = []

    for category in CATEGORIES:

        result = evaluate_category(
            category,
            device,
        )

        summary.append(
            result
        )

    df = pd.DataFrame(
        summary
    )

    output = (
        ROOT
        / "results"
        / "unet_summary.csv"
    )

    df.to_csv(
        output,
        index=False,
    )

    print(
        "\n\nU-NET SUMMARY\n"
    )

    print(df)

    print(
        f"\nSaved: {output}"
    )


if __name__ == "__main__":
    main()