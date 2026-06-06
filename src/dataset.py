from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class MVTecDataset(Dataset):
    """
    Dataset do MVTec AD.
    
    modo='train' -> só imagens normais (para treino unsupervised)
    modo='test'  -> normais + defeituosas com label e tipo de defeito
    """
    
    def __init__(self, root, category, modo='train', tamanho_imagem=256):
        self.root = Path(root)
        self.category = category
        self.modo = modo
        
        # Transformações: redimensiona, converte para tensor, normaliza
        self.transform = transforms.Compose([
            transforms.Resize((tamanho_imagem, tamanho_imagem)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # médias do ImageNet
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.imagens = []  # paths das imagens
        self.labels = []   # 0 = normal, 1 = defeituoso
        self.tipos = []    # nome do tipo de defeito
        
        self._carregar_imagens()
    
    def _carregar_imagens(self):
        if self.modo == 'train':
            # Só imagens normais
            pasta = self.root / self.category / "train" / "good"
            for img_path in sorted(pasta.glob("*.png")):
                self.imagens.append(img_path)
                self.labels.append(0)
                self.tipos.append("good")
        
        elif self.modo == 'test':
            pasta_test = self.root / self.category / "test"
            for tipo_pasta in sorted(pasta_test.iterdir()):
                label = 0 if tipo_pasta.name == "good" else 1
                for img_path in sorted(tipo_pasta.glob("*.png")):
                    self.imagens.append(img_path)
                    self.labels.append(label)
                    self.tipos.append(tipo_pasta.name)
    
    def __len__(self):
        return len(self.imagens)
    
    def __getitem__(self, idx):
        img = Image.open(self.imagens[idx]).convert("RGB")
        img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label, self.tipos[idx]


def criar_dataloaders(root, category, batch_size=32, tamanho_imagem=256, num_workers=0):
    """Cria dataloaders de treino e teste prontos a usar.

    num_workers=0 por omissão (seguro em Windows/Jupyter). Aumenta em Linux
    ou ao correr como script para acelerar o carregamento.
    """

    train_dataset = MVTecDataset(root, category, modo='train', tamanho_imagem=tamanho_imagem)
    test_dataset  = MVTecDataset(root, category, modo='test',  tamanho_imagem=tamanho_imagem)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    print(f"Categoria: {category}")
    print(f"  Treino: {len(train_dataset)} imagens normais")
    print(f"  Teste:  {len(test_dataset)} imagens ({sum(test_dataset.labels)} defeituosas)")
    
    return train_loader, test_loader


class ImagePathDataset(Dataset):
    """Dataset simples para uma lista explicita de imagens.

    Usado pelos scripts Bottle-only para criar splits reprodutiveis sem
    depender de estado escondido nos notebooks.
    """

    def __init__(self, image_paths, labels=None, tipos=None, tamanho_imagem=256):
        self.image_paths = [Path(p) for p in image_paths]
        self.labels = labels if labels is not None else [0] * len(self.image_paths)
        self.tipos = tipos if tipos is not None else ["good"] * len(self.image_paths)
        self.transform = transforms.Compose([
            transforms.Resize((tamanho_imagem, tamanho_imagem)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label, self.tipos[idx], str(self.image_paths[idx])


def criar_loader_de_paths(
    image_paths,
    labels=None,
    tipos=None,
    batch_size=32,
    tamanho_imagem=256,
    shuffle=False,
    num_workers=0,
):
    dataset = ImagePathDataset(
        image_paths=image_paths,
        labels=labels,
        tipos=tipos,
        tamanho_imagem=tamanho_imagem,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
