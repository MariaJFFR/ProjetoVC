import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from bottle_protocol import bottle_root, bottle_test_records, bottle_train_good_paths
from utils import save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train/evaluate illustrative Bottle CNN baseline.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.4)
    return parser.parse_args()


def load_image(path, image_size):
    img = Image.open(path).convert("RGB").resize((image_size, image_size))
    return np.asarray(img, dtype=np.float32) / 255.0


def build_model(image_size):
    from keras.layers import Conv2D, Dense, Flatten, Input, MaxPool2D
    from keras.models import Sequential

    model = Sequential([
        Input(shape=(image_size, image_size, 3)),
        Conv2D(filters=32, kernel_size=(3, 3), activation="relu"),
        MaxPool2D(pool_size=(2, 2)),
        Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
        MaxPool2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def main():
    args = parse_args()
    set_seed(args.seed)

    import tensorflow as tf

    tf.keras.utils.set_random_seed(args.seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    # Official test defects are the only Bottle defect labels available in MVTec.
    # This split is therefore illustrative and not an official benchmark protocol.
    official_test = bottle_test_records()
    test_paths = np.asarray([r.path for r in official_test], dtype=object)
    test_labels = np.asarray([r.label for r in official_test], dtype=int)
    supervised_train_paths, eval_paths, supervised_train_labels, eval_labels = train_test_split(
        test_paths,
        test_labels,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=test_labels,
    )

    normal_train_paths = np.asarray(bottle_train_good_paths(), dtype=object)
    normal_train_labels = np.zeros(len(normal_train_paths), dtype=int)
    train_paths = np.concatenate([normal_train_paths, supervised_train_paths])
    train_labels = np.concatenate([normal_train_labels, supervised_train_labels])

    x_train = np.stack([load_image(p, args.image_size) for p in train_paths])
    y_train = train_labels.astype(np.float32)
    x_eval = np.stack([load_image(p, args.image_size) for p in eval_paths])
    y_eval = eval_labels.astype(np.float32)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=train_labels,
    )
    class_weight = {0: float(class_weights[0]), 1: float(class_weights[1])}

    model = build_model(args.image_size)
    history = model.fit(
        x_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.1,
        class_weight=class_weight,
        verbose=1,
    )

    probs = model.predict(x_eval, verbose=0).ravel()
    preds = (probs >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_eval, preds)),
        "precision": float(precision_score(y_eval, preds, zero_division=0)),
        "recall": float(recall_score(y_eval, preds, zero_division=0)),
        "f1": float(f1_score(y_eval, preds, zero_division=0)),
        "auroc": float(roc_auc_score(y_eval, probs)),
    }

    model_path = ROOT / "models" / "cnn_bottle.keras"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)

    output = {
        "category": "bottle",
        "model": "cnn_supervised_illustrative",
        "protocol": (
            "Illustrative supervised baseline. Defect samples come from the official "
            "MVTec Bottle test set, so this is not directly comparable to the official "
            "anomaly-detection protocol."
        ),
        "seed": args.seed,
        "image_size": args.image_size,
        "train_count": int(len(train_paths)),
        "eval_count": int(len(eval_paths)),
        "official_train_good_count_used": int(len(normal_train_paths)),
        "official_test_samples_used_for_supervised_training": int(len(supervised_train_paths)),
        "metrics": metrics,
        "class_weight": class_weight,
        "model_path": str(model_path),
        "history": {
            key: [float(v) for v in values]
            for key, values in history.history.items()
        },
    }
    out_path = ROOT / "results" / "cnn_bottle_metrics.json"
    save_json(output, out_path)
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {out_path}")
    print(output)


if __name__ == "__main__":
    main()
