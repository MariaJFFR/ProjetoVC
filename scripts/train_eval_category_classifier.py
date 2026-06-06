from pathlib import Path

import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = ROOT / "data" / "category_classifier"

IMG_SIZE = (224, 224)

BATCH_SIZE = 32

SEED = 42

EPOCHS = 15


def main():

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    class_names = train_ds.class_names

    print("\nClasses:")
    print(class_names)

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )

    base_model.trainable = False

    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1.0 / 255),

        base_model,

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(
            128,
            activation="relu"
        ),

        tf.keras.layers.Dense(
            len(class_names),
            activation="softmax"
        ),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
    )

    model.save(
        ROOT /
        "models" /
        "category_classifier.keras"
    )

    y_true = []
    y_pred = []

    for images, labels in val_ds:

        predictions = model.predict(
            images,
            verbose=0
        )

        preds = predictions.argmax(axis=1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds)

    print("\nClassification Report\n")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names
        )
    )

    print("\nConfusion Matrix\n")

    print(
        confusion_matrix(
            y_true,
            y_pred
        )
    )

    loss, accuracy = model.evaluate(
        val_ds,
        verbose=0
    )

    print(f"\nValidation Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()