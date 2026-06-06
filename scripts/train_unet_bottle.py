import sys

from pathlib import Path

import numpy as np

import torch

import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data import random_split

ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(ROOT / "src"))

from unet import UNet
from unet_dataset import defective_records_only
from unet_dataset import BottleSegmentationDataset

from utils import get_device
from utils import set_seed


SEED = 42
BATCH_SIZE = 4
EPOCHS = 30
IMAGE_SIZE = 256


def main():

    set_seed(SEED)

    device = get_device()

    records = defective_records_only()

    dataset = BottleSegmentationDataset(
        records,
        image_size=IMAGE_SIZE
    )

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_ds, val_ds, test_ds = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE
    )

    model = UNet().to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4
    )

    best_loss = float("inf")

    for epoch in range(EPOCHS):

        model.train()

        train_loss = 0

        for images, masks in train_loader:

            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, masks)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        model.eval()

        val_loss = 0

        with torch.no_grad():

            for images, masks in val_loader:

                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)

                loss = criterion(outputs, masks)

                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        print(
            f"Epoch {epoch+1}/{EPOCHS}"
            f" Train={train_loss:.4f}"
            f" Val={val_loss:.4f}"
        )

        if val_loss < best_loss:

            best_loss = val_loss

            torch.save(
                model.state_dict(),
                ROOT / "models" / "unet_bottle_best.pth"
            )

    torch.save(
        test_ds,
        ROOT / "models" / "unet_test_split.pt"
    )


if __name__ == "__main__":
    main()