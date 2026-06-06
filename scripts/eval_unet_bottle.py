import sys

from pathlib import Path

import numpy as np

import torch

ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(ROOT / "src"))

from unet import UNet

from metrics import (
    segmentation_metrics,
    safe_auroc
)

from utils import (
    get_device,
    save_json
)


def main():

    device = get_device()

    test_ds = torch.load(
        ROOT / "models" / "unet_test_split.pt"
    )

    model = UNet().to(device)

    model.load_state_dict(
        torch.load(
            ROOT / "models" / "unet_bottle_best.pth",
            map_location=device
        )
    )

    model.eval()

    pixel_scores = []
    pixel_labels = []

    with torch.no_grad():

        for image, mask in test_ds:

            image = image.unsqueeze(0).to(device)

            pred = torch.sigmoid(
                model(image)
            )

            pixel_scores.extend(
                pred.cpu().numpy().ravel()
            )

            pixel_labels.extend(
                mask.numpy().ravel()
            )

    pixel_scores = np.asarray(pixel_scores)
    pixel_labels = np.asarray(pixel_labels)

    threshold = 0.5

    metrics = segmentation_metrics(
        pixel_labels,
        pixel_scores,
        threshold
    )

    metrics["pixel_auroc"] = safe_auroc(
        pixel_labels,
        pixel_scores
    )

    output = {
        "category": "bottle",
        "model": "unet",
        "metrics": metrics
    }

    save_json(
        output,
        ROOT / "results" / "unet_bottle_metrics.json"
    )

    print(output)


if __name__ == "__main__":
    main()