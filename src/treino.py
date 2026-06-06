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


def avaliar_loss_reconstrucao(modelo, data_loader, criterio, device='cpu'):
    modelo.eval()
    loss_total = 0.0
    with torch.no_grad():
        for batch in data_loader:
            imgs = batch[0].to(device)
            reconstrucao = modelo(imgs)
            loss = criterio(reconstrucao, imgs)
            loss_total += loss.item()
    return loss_total / max(1, len(data_loader))


def treinar_autoencoder_com_validacao(
    modelo,
    train_loader,
    val_loader,
    num_epochs=50,
    lr=1e-3,
    device='cpu',
    caminho_modelo="../models/autoencoder_bottle_best.pth",
):
    """Treina o autoencoder e guarda o checkpoint com menor loss de validacao.

    A validacao usa apenas imagens normais de bottle/train/good separadas do
    treino. Isso evita escolher o checkpoint com base no conjunto final de teste.
    """
    import os

    os.makedirs(os.path.dirname(caminho_modelo), exist_ok=True)

    modelo = modelo.to(device)
    optimizer = torch.optim.Adam(modelo.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=5,
        factor=0.5,
    )
    criterio = nn.MSELoss()

    historico = []
    melhor_val_loss = float('inf')

    for epoch in range(num_epochs):
        modelo.train()
        train_loss_total = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            imgs = batch[0].to(device)
            reconstrucao = modelo(imgs)
            loss = criterio(reconstrucao, imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()

        train_loss = train_loss_total / max(1, len(train_loader))
        val_loss = avaliar_loss_reconstrucao(modelo, val_loader, criterio, device)
        scheduler.step(val_loss)

        historico.append({
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
        })

        if val_loss < melhor_val_loss:
            melhor_val_loss = val_loss
            torch.save(modelo.state_dict(), caminho_modelo)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"train loss: {train_loss:.4f} | "
                f"val loss: {val_loss:.4f} | "
                f"best val: {melhor_val_loss:.4f}"
            )

    print(f"\nTreino concluido. Melhor val loss: {melhor_val_loss:.4f}")
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
