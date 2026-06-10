from pathlib import Path
import sys
import tempfile

import joblib
import numpy as np
import streamlit as st
import tensorflow as tf
import torch

from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parent

SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval_resnet import (
    build_resnet18_feature_extractor,
    extract_features,
    mahalanobis_scores,
)

from unet import UNet
from utils import get_device


CLASS_NAMES = [
    "bottle",
    "cable",
    "capsule",
    "tile",
    "wood",
]

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

def create_overlay(image_path, mask):

    image = Image.open(image_path).convert("RGB")
    image = image.resize((256, 256))

    image_np = np.asarray(image)

    binary_mask = (mask > 0.2).astype(np.uint8)

    overlay = image_np.copy()

    overlay[
        binary_mask == 1
    ] = (
        0.7 * overlay[binary_mask == 1]
        + 0.3 * np.array([0, 0, 255])
    ).astype(np.uint8)

    return overlay

def main():

    st.set_page_config(
        page_title="Industrial Defect Detection",
        layout="wide",
    )

    st.title(
        "Industrial Defect Detection"
    )

    uploaded_file = st.file_uploader(
        "Upload image",
        type=["png", "jpg", "jpeg"],
    )

    if uploaded_file is None:
        return

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Input Image",
        width=400,
    )

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".png",
    ) as tmp:

        tmp.write(uploaded_file.getbuffer())

        image_path = tmp.name

    if st.button("Run Pipeline"):

        with st.spinner("Running..."):

            category, confidence = (
                predict_category(
                    image_path
                )
            )

            anomaly, score, threshold = (
                detect_anomaly(
                    image_path,
                    category,
                )
            )

        st.subheader("Classification")

        st.write(
            f"Category: {category}"
        )

        st.write(
            f"Confidence: {confidence:.4f}"
        )

        st.subheader("Anomaly Detection")

        st.write(
            f"Anomaly Score: {score:.4f}"
        )

        st.write(
            f"Threshold: {threshold:.4f}"
        )

        if anomaly:

            st.error("DEFECTIVE")

            mask = segment_defect(
                image_path,
                category,
            )
            print("Mask min:", mask.min())
            print("Mask max:", mask.max())
            print("Mask mean:", mask.mean())

            overlay = create_overlay(
                image_path,
                mask,
            )
            
            mask_display = (
                mask - mask.min()
            ) / (
                mask.max() - mask.min() + 1e-8
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.image(
                    image,
                    caption="Original",
                )

            st.image(
                mask_display,
                caption=f"Mask (max={mask.max():.3f})"
            )

            with col3:
                st.image(
                    overlay,
                    caption="Overlay",
                )

        else:

            st.success("NORMAL")


if __name__ == "__main__":
    main()
