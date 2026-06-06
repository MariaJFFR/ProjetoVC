from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf

import torch

import sys

from PIL import Image

import matplotlib.pyplot as plt

from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "src")
)

from unet import UNet

ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(ROOT / "src"))

from eval_resnet import (
    build_resnet18_feature_extractor,
    extract_features,
    mahalanobis_scores,
)

from utils import get_device


CLASS_NAMES = [
    "bottle",
    "cable",
    "capsule",
    "tile",
    "wood",
]

def segment_defect(
    image_path,
    category,
):

    device = get_device()

    model = UNet().to(device)

    model.load_state_dict(
        torch.load(
            ROOT
            / "models"
            / f"unet_{category}_best.pth",
            map_location=device,
        )
    )

    model.eval()

    tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        )
    ])

    image = Image.open(
        image_path
    ).convert("RGB")

    x = tf(image).unsqueeze(0).to(device)

    with torch.no_grad():

        mask = torch.sigmoid(
            model(x)
        )

    mask = (
        mask
        .squeeze()
        .cpu()
        .numpy()
    )

    return mask

def save_segmentation_results(
    image_path,
    category,
    mask,
):

    output_dir = (
        ROOT
        / "results"
        / "predictions"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    image = Image.open(
        image_path
    ).convert("RGB")

    image = image.resize(
        (256, 256)
    )

    image_np = np.asarray(
        image
    )

    binary_mask = (
        mask > 0.5
    ).astype(np.uint8)

    overlay = image_np.copy()

    overlay[
        binary_mask == 1
    ] = [255, 0, 0]

    stem = Path(
        image_path
    ).stem

    mask_path = (
        output_dir
        / f"{category}_{stem}_mask.png"
    )

    overlay_path = (
        output_dir
        / f"{category}_{stem}_overlay.png"
    )

    plt.imsave(
        mask_path,
        binary_mask,
        cmap="gray",
    )

    plt.imsave(
        overlay_path,
        overlay,
    )

    return (
        mask_path,
        overlay_path,
    )

def predict_category(image_path):

    model = tf.keras.models.load_model(
        ROOT /
        "models" /
        "category_classifier.keras"
    )

    image = tf.keras.utils.load_img(
        image_path,
        target_size=(224, 224)
    )

    image = tf.keras.utils.img_to_array(
        image
    )

    image = np.expand_dims(
        image,
        axis=0
    )

    probs = model.predict(
        image,
        verbose=0
    )[0]

    idx = np.argmax(probs)

    return (
        CLASS_NAMES[idx],
        float(probs[idx])
    )


def detect_anomaly(image_path, category):

    device = get_device()

    resnet, preprocess = (
        build_resnet18_feature_extractor(
            device
        )
    )

    model_path = (
        ROOT /
        "models" /
        f"resnet_mahalanobis_{category}.joblib"
    )

    model_data = joblib.load(
        model_path
    )

    estimator = model_data["estimator"]

    threshold = model_data["threshold"]

    features = extract_features(
        [image_path],
        resnet,
        preprocess,
        device,
    )

    score = float(
        mahalanobis_scores(
            estimator,
            features,
        )[0]
    )

    is_anomaly = score > threshold

    return (
        is_anomaly,
        score,
        threshold,
    )


def main():

    image_path = input(
        "Image path: "
    ).strip()

    print(
        "\nClassifying category..."
    )

    category, confidence = (
        predict_category(
            image_path
        )
    )

    print(
        f"Category: {category}"
    )

    print(
        f"Confidence: {confidence:.4f}"
    )

    print(
        "\nDetecting anomaly..."
    )

    anomaly, score, threshold = (
        detect_anomaly(
            image_path,
            category
        )
    )

    print(
        f"Anomaly Score: {score:.4f}"
    )

    print(
        f"Threshold: {threshold:.4f}"
    )

    if anomaly:

        print(
            "\nResult: DEFECTIVE"
        )

        print(
            "\nRunning U-Net..."
        )

        mask = segment_defect(
            image_path,
            category,
        )

        mask_path, overlay_path = (
            save_segmentation_results(
                image_path,
                category,
                mask,
            )
        )

        print(
            f"Mask saved: {mask_path}"
        )

        print(
            f"Overlay saved: {overlay_path}"
        )

    else:

        print(
            "\nResult: NORMAL"
        )


if __name__ == "__main__":
    main()