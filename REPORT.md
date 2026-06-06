# Technical Report - Bottle-Only Refactor

## 1. Current Repository Architecture

The project is a Bottle-only MVTec AD experiment suite with:

- `src/dataset.py`: PyTorch dataset/dataloader helpers.
- `src/autoencoder.py`: convolutional autoencoder.
- `src/treino.py`: autoencoder training functions.
- `src/bottle_protocol.py`: Bottle-only paths, splits, and mask lookup.
- `src/metrics.py`: standardized classification and segmentation metrics.
- `src/eval_autoencoder.py`: reconstruction-error and mask helpers.
- `src/eval_resnet.py`: ResNet18 feature and Mahalanobis helpers.
- `scripts/`: runnable Bottle-only training/evaluation commands.
- `notebooks/`: original exploratory experiments.
- `models/`: saved model artifacts.
- `results/`: saved metrics and histories.

## 2. Problems Discovered

The original notebooks were useful exploratory experiments, but several details
made the reported metrics scientifically fragile:

- Autoencoder checkpoint selection used training loss only.
- Autoencoder and ResNet thresholds were selected from final test labels or test
  score distributions.
- Pixel threshold selection in `05_metricas.ipynb` maximized Dice on final test
  masks, which leaks ground truth into the reported operating point.
- Segmentation masks were resized with PIL's default interpolation instead of
  nearest-neighbor interpolation.
- The CNN mixed official MVTec test images into training and then reported a
  random-split score as if it were comparable to anomaly-detection results.
- Random seeds were not fixed.
- Metrics were printed in notebooks but not saved consistently.
- `models/`, `results/`, and `requirements.txt` were missing.
- Evaluation logic was duplicated across notebooks.
- The project contained all MVTec categories on disk, but the implemented
  experiments were Bottle-only.

### File-by-File Audit

| File | Purpose | Issues Found | Refactor Status |
|---|---|---|---|
| `README.md` | Project overview and execution notes | Reported metrics without enough protocol caveats; missing reproducible script order | Rewritten |
| `.gitignore` | Ignore large data and artifacts | Did not cover new model/result artifact types | Updated |
| `src/dataset.py` | Original PyTorch MVTec dataloader | Image-level only; no explicit path-based split support | Kept compatible and extended with path dataloader |
| `src/autoencoder.py` | Autoencoder model | Architecture is consistent with current scope | Preserved |
| `src/treino.py` | Autoencoder training | Best model selected by train loss only | Validation-aware trainer added |
| `notebooks/00_teste_dataloaders.ipynb` | Loader sanity check | Exploratory only; no mask validation | Kept as exploratory notebook |
| `notebooks/01_exploração.ipynb` | Dataset exploration | Saves PNG artifact; not part of reproducible protocol | Kept as exploratory notebook |
| `notebooks/02_autoencoder.ipynb` | Original autoencoder experiment | No validation split; checkpoint selected by train loss | Replaced for reproducible use by scripts |
| `notebooks/03_classificador.ipynb` | Original CNN experiment | Random split mixes official test data; benchmark comparison is invalid | Replaced for reproducible use by documented CNN script |
| `notebooks/04_resnet.ipynb` | Original ResNet Mahalanobis experiment | Threshold selected from test ROC labels | Replaced for reproducible use by script |
| `notebooks/05_metricas.ipynb` | Original metric notebook | Pixel threshold maximizes test Dice; default mask interpolation | Replaced for reproducible use by script |
| `scripts/train_autoencoder_bottle.py` | Train AE under corrected protocol | New | Added |
| `scripts/eval_autoencoder_bottle.py` | Evaluate AE with validation-calibrated thresholds | New | Added |
| `scripts/eval_resnet_bottle.py` | Evaluate ResNet Mahalanobis with validation-calibrated threshold | New | Added |
| `scripts/train_eval_cnn_bottle.py` | Illustrative supervised CNN baseline | New; explicitly non-comparable to official protocol | Added |

## 3. Corrections Applied

### Autoencoder

- Added deterministic split of `bottle/train/good` into train-normal and
  validation-normal subsets.
- Added validation-aware training in `treino.py`.
- Best checkpoint is now selected by validation reconstruction loss.
- Image threshold is calibrated on validation-normal image scores.
- Pixel threshold is calibrated on validation-normal pixel errors.
- Final image and pixel metrics are computed only on `bottle/test`.

### ResNet18 + Mahalanobis

- Feature covariance is fit only on train-normal Bottle images.
- Threshold is calibrated on validation-normal Mahalanobis scores.
- Final metrics are computed only on `bottle/test`.

### CNN

- Kept as an illustrative supervised baseline.
- The script explicitly records that official test defects are used for
  supervised learning, so the result is not directly comparable to the official
  MVTec anomaly-detection protocol.
- Added AUROC alongside accuracy, precision, recall, and F1.

### Metrics and Masks

- Added standardized metric helpers.
- Added nearest-neighbor mask resizing.
- Added JSON result saving.
- Added fixed random seeds and deterministic PyTorch/TensorFlow settings where
  possible.

## 4. New Execution Order

From the repository root:

```bash
pip install -r requirements.txt
python scripts/train_autoencoder_bottle.py
python scripts/eval_autoencoder_bottle.py
python scripts/eval_resnet_bottle.py
python scripts/train_eval_cnn_bottle.py
```

## 5. Folder Structure

```text
data/mvtec/bottle/       Bottle dataset and masks
models/                  Saved model artifacts
notebooks/               Original exploratory notebooks
results/                 JSON metrics and training histories
scripts/                 Reproducible Bottle-only commands
src/                     Shared Bottle-only code
```

## 6. Experimental Methodology

### Unsupervised Methods

The autoencoder and ResNet18 + Mahalanobis methods follow the one-class anomaly
detection protocol:

1. Use only normal Bottle training images.
2. Split `train/good` into train and validation subsets.
3. Fit/train on train-normal images.
4. Calibrate thresholds on validation-normal images.
5. Evaluate once on official Bottle test images.

This avoids using test labels or masks to choose thresholds.

### Supervised CNN

MVTec Bottle does not include defective training images. A supervised defect
classifier therefore cannot be trained under the official anomaly-detection
protocol without using official test defects or an external defect dataset.

The CNN remains in the project only as an illustrative supervised experiment.
Its metrics should not be compared directly with the unsupervised benchmark
metrics.

## 7. Remaining Limitations

- The autoencoder is a simple reconstruction baseline and localization quality
  is expected to be limited.
- Threshold calibration uses normal validation images only. This is realistic
  for anomaly detection, but the selected false-positive operating point is a
  design choice.
- The CNN baseline is not official-protocol comparable because defect training
  samples come from the official test set.
- No new anomaly-detection methods were added by design.
- The notebooks still contain the original exploratory code; scripts are the
  recommended reproducible path.
