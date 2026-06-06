from pathlib import Path

import numpy as np

import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT /
    "models" /
    "category_classifier.keras"
)

CLASS_NAMES = [
    "bottle",
    "cable",
    "capsule",
    "tile",
    "wood",
]


def predict_image(image_path):

    model = tf.keras.models.load_model(
        MODEL_PATH
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

    prediction = CLASS_NAMES[
        np.argmax(probs)
    ]

    print(
        f"Prediction: {prediction}"
    )

    print(
        f"Confidence: {probs.max():.4f}"
    )


if __name__ == "__main__":

    image_path = input(
        "Image path: "
    )

    predict_image(image_path)