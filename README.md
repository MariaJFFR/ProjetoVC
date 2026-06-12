# Deteção de Defeitos Industriais — MVTec AD

Projeto feito para a cadeira de visão por computador de inspeção visual de qualidade industrial sobre o dataset
**MVTec AD**. O objetivo é **detetar** e **localizar** defeitos em produtos,
comparando abordagens **supervisionadas** e **não supervisionadas**.

A categoria principal de estudo é a **`bottle`**; alguns modelos foram ainda
treinados em **5 categorias** (`bottle`, `cable`, `capsule`, `tile`, `wood`)
para avaliar a generalização.

---

## Modelos implementados

| Modelo | Tipo | Tarefa | Como funciona |
|---|---|---|---|
| **Autoencoder** | Não supervisionado | Deteção + localização | Treina só com imagens normais; o **erro de reconstrução** é o score de anomalia |
| **ResNet18 + Mahalanobis** | Não supervisionado | Deteção | Features pré-treinadas (ImageNet) + **distância de Mahalanobis** à distribuição normal |
| **CNN** | Supervisionado | Deteção | Classificador binário normal/defeito (*baseline ilustrativo*) |
| **U-Net** | Supervisionado | Localização | Segmentação ao pixel com máscaras de *ground truth* |
| **Classificador de tipo de defeito** | Supervisionado | Identificar o defeito | Features ResNet18 + LogisticRegression (por categoria) |

> O **pipeline** (`scripts/predict_pipeline.py`) junta tudo: classifica a
> categoria → deteta anomalia → se houver defeito, segmenta-o.

---

## Estrutura do projeto (arquitetura)

O código está organizado de forma **modular**, com um fluxo unidirecional:

```
data/  →  src/  →  scripts/  →  models/ + results/  →  notebooks/
(dados)  (lógica)  (execução)   (artefactos)          (visualização)
```

| Pasta | Conteúdo |
|---|---|
| `data/` | Dataset MVTec AD (imagens + máscaras). **Só leitura, nunca modificada.** |
| `src/` | Código reutilizável: modelos (`autoencoder.py`, `unet.py`), protocolos/splits (`bottle_protocol.py`, `category_protocol.py`), datasets (`dataset.py`, `unet_dataset.py`) e métricas (`metrics.py`, `eval_*.py`). |
| `scripts/` | Programas executáveis — um por modelo (treino + avaliação) + o pipeline. |
| `models/` | Modelos treinados (`.pth`, `.keras`, `.joblib`). |
| `results/` | Métricas e históricos (`.json`, `.csv`). |
| `notebooks/` | Exploração e **visualização** dos resultados (gráficos, exemplos, matrizes). |

A lógica vive em `src/`, por isso **scripts e notebooks usam o mesmo código** e
dão resultados idênticos.

---

## Dataset

O dataset **não está incluído** (é grande). Descarrega o
[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) e coloca-o
em `data/mvtec/`, com esta estrutura por categoria:

```text
data/mvtec/bottle/
  train/good/*.png                 # imagens normais (treino)
  test/good/*.png                  # imagens normais (teste)
  test/<tipo_de_defeito>/*.png      # imagens defeituosas (teste)
  ground_truth/<tipo>/*_mask.png    # máscaras dos defeitos
```

---

## Instalação

```bash
pip install -r requirements.txt
```

Requer Python 3.10+ com PyTorch, TensorFlow/Keras, scikit-learn, etc.
(GPU recomendada para o autoencoder e a U-Net, mas não obrigatória.)

---

## Como correr

Corre todos os comandos **a partir da raiz do projeto**.

```bash
# 1. Autoencoder (deteção + localização) — bottle
python scripts/train_autoencoder_bottle.py     # treina e guarda o modelo
python scripts/eval_autoencoder_bottle.py      # avalia (AUROC, IoU, Dice)

# 2. ResNet18 + Mahalanobis (deteção) — 5 categorias
python scripts/train_resnet_category.py

# 3. CNN supervisionada (baseline) — bottle
python scripts/train_eval_cnn_bottle.py

# 4. U-Net (segmentação) — 5 categorias
python scripts/train_unet_category.py          # treina
python scripts/eval_unet_category.py           # avalia (Pixel-AUROC, IoU, Dice)

# 5. Classificador de tipo de defeito (Tarefa 2) — 5 categorias
python scripts/train_eval_defect_classifier.py

# 6. (Opcional) Classificador de categoria, para o pipeline
python scripts/build_category_dataset.py
python scripts/train_eval_category_classifier.py
```

Cada script **guarda os modelos em `models/`** e as **métricas em `results/`**.

Pipeline end-to-end (interativo — pede o caminho de uma imagem):

```bash
python scripts/predict_pipeline.py
```

Depois de treinar/avaliar, abre os **notebooks 02–07** para ver os gráficos.

---

## Notebooks

| Notebook | Conteúdo |
|---|---|
| `00_teste_dataloaders` / `01_exploração` | Exploração do dataset |
| `02_autoencoder` | Autoencoder: treino + reconstruções |
| `03_classificador` | CNN: treino + matriz de confusão + exemplos |
| `04_resnet` | ResNet+Mahalanobis: scores + AUROC por categoria |
| `05_metricas` | Autoencoder: métricas de localização + falsos positivos |
| `06_tipo_defeito` | Classificador de tipo de defeito (Tarefa 2) |
| `07_unet` | U-Net: segmentação + comparação com o autoencoder |

Os notebooks reutilizam as funções de `src/`, por isso reproduzem **a mesma
metodologia** dos scripts.

---

## Metodologia (sem *data leakage*)

Princípio seguido em todos os modelos:

1. As imagens normais de treino são divididas em **treino** e **validação**.
2. O conjunto de **teste** é reservado só para a avaliação final.
3. A seleção do melhor modelo e a calibração dos **limiares** são feitas na
   **validação**, **nunca no teste**.
4. As *seeds* são fixadas e os resultados são guardados em ficheiros.

Isto evita a **fuga de informação** (métricas artificialmente otimistas) e
garante que os resultados são **reprodutíveis**.

> Nota: a CNN é um **baseline ilustrativo**. Como o MVTec não tem defeitos no
> treino, usa parte dos defeitos do conjunto de teste — por isso as suas
> métricas **não são diretamente comparáveis** às dos métodos não supervisionados.
