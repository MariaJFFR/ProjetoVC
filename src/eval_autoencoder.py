from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from bottle_protocol import mask_path_for_image


def reconstruction_maps(model, loader, device):
    criterion = torch.nn.MSELoss(reduction="none")
    image_scores = []
    labels = []
    defect_types = []
    paths = []
    error_maps = []

    model.eval()
    with torch.no_grad():
        for imgs, batch_labels, batch_types, batch_paths in loader:
            imgs = imgs.to(device)
            recon = model(imgs)
            err = criterion(recon, imgs)
            err_maps = err.mean(dim=1).cpu().numpy()
            scores = err.mean(dim=(1, 2, 3)).cpu().numpy()
            image_scores.extend(scores.tolist())
            labels.extend(batch_labels.numpy().astype(int).tolist())
            defect_types.extend(list(batch_types))
            paths.extend([Path(p) for p in batch_paths])
            error_maps.extend([m for m in err_maps])

    return {
        "image_scores": np.asarray(image_scores),
        "labels": np.asarray(labels),
        "defect_types": np.asarray(defect_types),
        "paths": paths,
        "error_maps": error_maps,
    }


def load_mask_for_path(path, size, data_root=None):
    mask_path = mask_path_for_image(path, data_root)
    if mask_path is None:
        return np.zeros((size, size), dtype=np.uint8)
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing mask for {path}: {mask_path}")
    nearest = getattr(Image, "Resampling", Image).NEAREST
    mask = Image.open(mask_path).convert("L").resize(
        (size, size),
        resample=nearest,
    )
    return (np.asarray(mask) > 0).astype(np.uint8)


def flatten_test_masks(paths, size, data_root=None):
    masks = [load_mask_for_path(path, size, data_root).ravel() for path in paths]
    return np.concatenate(masks)


def validation_pixel_errors(model, paths, device, image_size=256):
    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    all_pixels = []
    model.eval()
    with torch.no_grad():
        for path in paths:
            img = Image.open(path).convert("RGB")
            x = tf(img).unsqueeze(0).to(device)
            rec = model(x)
            err = ((rec - x) ** 2).mean(dim=1).squeeze(0).cpu().numpy()
            all_pixels.append(err.ravel())
    return np.concatenate(all_pixels)
