import numpy as np
import torch
from PIL import Image
from sklearn.covariance import LedoitWolf
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18


def build_resnet18_feature_extractor(device):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return model, preprocess


def extract_features(paths, model, preprocess, device):
    feats = []
    with torch.no_grad():
        for path in paths:
            img = Image.open(path).convert("RGB")
            x = preprocess(img).unsqueeze(0).to(device)
            feats.append(model(x).squeeze(0).cpu().numpy())
    return np.asarray(feats)


def fit_mahalanobis(train_features):
    return LedoitWolf().fit(train_features)


def mahalanobis_scores(estimator, features):
    return estimator.mahalanobis(features)
