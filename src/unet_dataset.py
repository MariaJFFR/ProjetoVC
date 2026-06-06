from pathlib import Path

import numpy as np
import torch

from PIL import Image

from torch.utils.data import Dataset

from torchvision import transforms

from bottle_protocol import (
    bottle_test_records,
    mask_path_for_image,
)


class BottleSegmentationDataset(Dataset):

    def __init__(self, records, image_size=256):

        self.records = records

        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
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

        image = Image.open(record.path).convert("RGB")

        mask_path = mask_path_for_image(record.path)

        mask = Image.open(mask_path).convert("L")

        image = self.image_transform(image)

        mask = self.mask_transform(mask)

        mask = (mask > 0).float()

        return image, mask


def defective_records_only():

    records = bottle_test_records()

    return [
        r
        for r in records
        if r.label == 1
    ]