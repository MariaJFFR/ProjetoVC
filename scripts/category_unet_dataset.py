from pathlib import Path

import torch

from PIL import Image

from torch.utils.data import Dataset

from torchvision import transforms

from category_protocol import (
    category_test_records,
)

ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = ROOT / "data" / "mvtec"


def mask_path_for_image(category, image_path):

    image_path = Path(image_path)

    defect_type = image_path.parent.name

    filename = image_path.stem + "_mask.png"

    return (
        DATASET_ROOT
        / category
        / "ground_truth"
        / defect_type
        / filename
    )


class CategorySegmentationDataset(Dataset):

    def __init__(
        self,
        records,
        category,
        image_size=256,
    ):

        self.records = records
        self.category = category

        self.image_transform = transforms.Compose([
            transforms.Resize(
                (image_size, image_size)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.NEAREST
            ),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):

        record = self.records[idx]

        image = Image.open(
            record.path
        ).convert("RGB")

        mask_path = mask_path_for_image(
            self.category,
            record.path
        )

        mask = Image.open(
            mask_path
        ).convert("L")

        image = self.image_transform(
            image
        )

        mask = self.mask_transform(
            mask
        )

        mask = (mask > 0).float()

        return image, mask


def defective_records_only(category):

    records = category_test_records(
        category
    )

    return [
        r
        for r in records
        if r.label == 1
    ]