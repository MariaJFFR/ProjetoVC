from dataclasses import dataclass
from pathlib import Path

import numpy as np

from utils import SEED, project_root


CATEGORY = "bottle"
DEFECT_TYPES = ["broken_large", "broken_small", "contamination"]


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: int
    defect_type: str


def bottle_root(data_root=None):
    if data_root is None:
        data_root = project_root() / "data" / "mvtec"
    return Path(data_root) / CATEGORY


def bottle_train_good_paths(data_root=None):
    root = bottle_root(data_root)
    return sorted((root / "train" / "good").glob("*.png"))


def split_train_validation_good(paths, val_fraction=0.2, seed=SEED):
    """Deterministic normal-only split for unsupervised calibration."""
    paths = np.array([Path(p) for p in paths], dtype=object)
    rng = np.random.default_rng(seed)
    indices = np.arange(len(paths))
    rng.shuffle(indices)
    val_size = max(1, int(round(len(paths) * val_fraction)))
    val_idx = np.sort(indices[:val_size])
    train_idx = np.sort(indices[val_size:])
    return list(paths[train_idx]), list(paths[val_idx])


def bottle_test_records(data_root=None):
    root = bottle_root(data_root)
    records = []
    test_dir = root / "test"
    for sub in sorted(test_dir.iterdir()):
        if not sub.is_dir():
            continue
        label = 0 if sub.name == "good" else 1
        for path in sorted(sub.glob("*.png")):
            records.append(ImageRecord(path=path, label=label, defect_type=sub.name))
    return records


def mask_path_for_image(image_path, data_root=None):
    image_path = Path(image_path)
    defect_type = image_path.parent.name
    if defect_type == "good":
        return None
    return bottle_root(data_root) / "ground_truth" / defect_type / f"{image_path.stem}_mask.png"
