from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]

MVTEC = ROOT / "data" / "mvtec"
OUTPUT = ROOT / "data" / "category_classifier"

CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "tile",
    "wood",
]

OUTPUT.mkdir(parents=True, exist_ok=True)

for category in CATEGORIES:

    target_dir = OUTPUT / category
    target_dir.mkdir(parents=True, exist_ok=True)

    train_good = MVTEC / category / "train" / "good"
    test_good = MVTEC / category / "test" / "good"

    copied = 0

    for source_dir in [train_good, test_good]:

        for image_path in source_dir.glob("*.png"):

            destination = target_dir / f"{category}_{image_path.name}"

            shutil.copy2(image_path, destination)

            copied += 1

    print(f"{category}: {copied} images")

print("\nDataset created successfully.")