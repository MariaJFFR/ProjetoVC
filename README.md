# ProjetoVC — Deteção de anomalias (MVTec AD)

Project 2: inspeção visual de qualidade industrial. Categoria usada: `bottle`.
Implementámos **3 modelos** e comparámos abordagem **supervisionada vs não supervisionada**.

## Modelos

| Modelo | Tipo | Framework | Resultado |
|--------|------|-----------|-----------|
| Autoencoder convolucional | não supervisionado (treina só com imagens normais) | PyTorch | Image-AUROC **0.92** |
| CNN classifier | supervisionado (normal vs defeito) | Keras | Accuracy **0.93** |
| ResNet + Mahalanobis | não supervisionado (features pré-treinadas, sem treino) | PyTorch | Image-AUROC **~1.00** |

## Métricas (autoencoder)

- Image-level AUROC: **0.92**
- Pixel-level AUROC: **0.80**
- IoU / Dice (localização): **0.18 / 0.31**
- Falsos positivos: ~1/20 nas imagens normais

## Notebooks

- `00_teste_dataloaders` — teste do carregamento de dados
- `01_exploração` — exploração do dataset
- `02_autoencoder` — modelo 1 (não supervisionado)
- `03_classificador` — modelo 2 (supervisionado)
- `04_resnet` — modelo 3 (ResNet pré-treinada + Mahalanobis)
- `05_metricas` — pixel-AUROC, IoU/Dice, falsos positivos + **comparação final dos 3 modelos**

## Como correr

Requer `torch`, `torchvision`, `tensorflow`/`keras`, `scikit-learn`, `matplotlib`, `pillow`.
Abrir os notebooks pela ordem (00 → 05) e correr as células. O dataset MVTec deve estar em `data/mvtec/`.
