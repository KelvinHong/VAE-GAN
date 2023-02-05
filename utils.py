import torchvision
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader



class Encoder(nn.Module):
    def __init__(self, latent_dim):
        # Encoder for VAE
        super().__init__()

        self.conv1 = nn.Conv2d(1, 4, 3)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(4, 16, 3)
        self.conv3 = nn.Conv2d(16, 32, 3, 2)
        self.fc = nn.Linear(128, latent_dim)

    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        output = self.fc(x)
        return output

class VariationalEncoder(nn.Module):
    def __init__(self, latent_dim):
        # Encoder for VAE
        super().__init__()

        self.conv1 = nn.Conv2d(1, 4, 3)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(4, 16, 3)
        self.conv3 = nn.Conv2d(16, 32, 3, 2)
        self.mean_fc = nn.Linear(128, latent_dim)
        self.var_fc = nn.Linear(128, latent_dim)

    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        mean = self.mean_fc(x)
        var = torch.exp(self.var_fc(x))
        # Reparameterization trick
        epsilon = torch.normal(0, 1, size=mean.shape)
        z = mean + var * epsilon
        return z

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 3*3*32),
            nn.ReLU(True),
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32,3,3))
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)
        )
    
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x

class AutoEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
    
    def forward(self, x):
        z = self.encoder(x)
        output = self.decoder(z)
        return output

class VAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.vencoder = VariationalEncoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        z = self.vencoder(x)
        output = self.decoder(z)
        return output