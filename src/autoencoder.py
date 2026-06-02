import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder: comprime a imagem 256x256 até 8x8
        self.encoder = nn.Sequential(
            # 3 x 256 x 256 -> 32 x 128 x 128
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # 32 x 128 x 128 -> 64 x 64 x 64
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 64 x 64 x 64 -> 128 x 32 x 32
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # 128 x 32 x 32 -> 256 x 16 x 16
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # 256 x 16 x 16 -> 512 x 8 x 8
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        
        # Decoder: reconstrói a imagem a partir do código comprimido
        self.decoder = nn.Sequential(
            # 512 x 8 x 8 -> 256 x 16 x 16
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # 256 x 16 x 16 -> 128 x 32 x 32
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # 128 x 32 x 32 -> 64 x 64 x 64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 64 x 64 x 64 -> 32 x 128 x 128
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # 32 x 128 x 128 -> 3 x 256 x 256
            # Output linear (sem Tanh): as imagens estão normalizadas com as
            # médias/desvios do ImageNet, logo os valores caem fora de (-1, 1).
            # Um Tanh aqui limitaria a reconstrução e estragava o erro/AUROC.
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
        )
    
    def forward(self, x):
        codigo = self.encoder(x)
        reconstrucao = self.decoder(codigo)
        return reconstrucao