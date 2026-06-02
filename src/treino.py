import torch
import torch.nn as nn
from tqdm import tqdm


def treinar_autoencoder(modelo, train_loader, num_epochs=50, learning_rate=1e-3, device='cpu'):
    """Treino simples do autoencoder (MSE)."""
    modelo = modelo.to(device)
    optimizer = torch.optim.Adam(modelo.parameters(), lr=learning_rate)
    criterio = nn.MSELoss()
    historico = []

    for epoch in range(num_epochs):
        modelo.train()
        loss_total = 0

        for imgs, _, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            imgs = imgs.to(device)

            # Forward pass
            reconstrucao = modelo(imgs)
            loss = criterio(reconstrucao, imgs)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_total += loss.item()

        loss_media = loss_total / len(train_loader)
        historico.append(loss_media)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} — Loss: {loss_media:.4f}")

    return historico


def treinar_melhor(modelo, train_loader, num_epochs=50, lr=1e-3, device='cpu',
                   caminho_modelo="../models/autoencoder_bottle_best.pth"):
    """Treino com scheduler de learning rate e guarda automaticamente o melhor modelo."""
    import os

    os.makedirs(os.path.dirname(caminho_modelo), exist_ok=True)

    modelo = modelo.to(device)
    optimizer = torch.optim.Adam(modelo.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterio = nn.MSELoss()

    historico = []
    melhor_loss = float('inf')

    for epoch in range(num_epochs):
        modelo.train()
        loss_total = 0

        for imgs, _, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            imgs = imgs.to(device)
            reconstrucao = modelo(imgs)
            loss = criterio(reconstrucao, imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_total += loss.item()

        loss_media = loss_total / len(train_loader)
        historico.append(loss_media)
        scheduler.step(loss_media)

        # Guarda automaticamente quando melhora
        if loss_media < melhor_loss:
            melhor_loss = loss_media
            torch.save(modelo.state_dict(), caminho_modelo)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} — Loss: {loss_media:.4f} | Melhor: {melhor_loss:.4f}")

    print(f"\nTreino concluído! Melhor loss: {melhor_loss:.4f}")
    return historico
