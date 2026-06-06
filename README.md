# ProjetoVC - Bottle Anomaly Detection on MVTec AD

This project studies visual anomaly detection for the **Bottle** category of
MVTec AD. The scope is intentionally limited to Bottle and to the three methods
already present in the original project:

1. Convolutional autoencoder anomaly detection.
2. ResNet18 feature extraction + Mahalanobis anomaly scoring.
3. Supervised CNN classifier as an illustrative baseline.

The refactored protocol fixes the main experimental issues in the notebooks:
test-set threshold tuning, missing validation calibration, mask interpolation
errors, missing reproducibility controls, and inconsistent metric saving.

## Scope

This repository does **not** implement a generic MVTec pipeline. It does not add
support for all categories, category classifiers, PatchCore, PaDiM, U-Net, ViTs,
CLIP, or any other additional anomaly detection method.

## Dataset

Expected dataset location:

```text
data/mvtec/bottle/
  train/good/*.png
  test/good/*.png
  test/broken_large/*.png
  test/broken_small/*.png
  test/contamination/*.png
  ground_truth/<defect_type>/*_mask.png
```

The unsupervised methods use only `bottle/train/good` for fitting and threshold
calibration. `bottle/test` is reserved for final evaluation.

## Setup

```bash
pip install -r requirements.txt
```

## Execution Order

Run from the repository root:

```bash
python scripts/train_autoencoder_bottle.py
python scripts/eval_autoencoder_bottle.py
python scripts/eval_resnet_bottle.py
python scripts/train_eval_cnn_bottle.py
```

Generated files:

```text
models/autoencoder_bottle_best.pth
models/resnet_mahalanobis_bottle.joblib
models/cnn_bottle.keras
results/autoencoder_bottle_training.json
results/autoencoder_bottle_metrics.json
results/resnet_bottle_metrics.json
results/cnn_bottle_metrics.json
```

## Protocol Summary

### Autoencoder

- Train on a deterministic subset of `bottle/train/good`.
- Validate on the remaining normal images from `bottle/train/good`.
- Save the checkpoint with the best validation reconstruction loss.
- Calibrate image threshold from validation normal image reconstruction scores.
- Calibrate pixel threshold from validation normal reconstruction-error pixels.
- Evaluate only on `bottle/test`.

Metrics:

- Image AUROC
- Pixel AUROC
- Precision
- Recall
- F1
- IoU
- Dice

### ResNet18 + Mahalanobis

- Extract ImageNet-pretrained ResNet18 features.
- Fit LedoitWolf covariance on train-normal Bottle features.
- Calibrate threshold on validation-normal Mahalanobis scores.
- Evaluate only on `bottle/test`.

Metrics:

- Image AUROC
- Precision
- Recall
- F1

### Supervised CNN

The CNN is kept as an **illustrative supervised baseline**. MVTec Bottle does
not provide defective training images, so the CNN must use a split of the
official test defects to learn the defect class. Because of that, its metrics
are **not directly comparable** to the official MVTec anomaly-detection
protocol used by the unsupervised methods.

Metrics:

- Accuracy
- Precision
- Recall
- F1
- AUROC

## Notebooks

The notebooks remain useful as exploratory material:

- `00_teste_dataloaders.ipynb`: dataloader sanity check.
- `01_exploração.ipynb`: dataset exploration.
- `02_autoencoder.ipynb`: original autoencoder experiment.
- `03_classificador.ipynb`: original supervised CNN experiment.
- `04_resnet.ipynb`: original ResNet + Mahalanobis experiment.
- `05_metricas.ipynb`: original metric exploration.

For reproducible results, prefer the scripts under `scripts/`.
