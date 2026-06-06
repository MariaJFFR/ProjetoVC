import json
import random
from pathlib import Path

import numpy as np


SEED = 42


def project_root():
    return Path(__file__).resolve().parents[1]


def set_seed(seed=SEED, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def get_device():
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data, path):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)
