import torchvision
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader



class Encoder(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=2, padding=1)
        self.batch2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 3, stride=2, padding=0)
        self.linear1 = nn.Linear(3*3*32, 128)
        self.linear2 = nn.Linear(128, latent_dims) # generate latent 
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.batch2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        z = self.linear2(x)
        return z

class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=2, padding=1)
        self.batch2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 3, stride=2, padding=0)
        self.linear1 = nn.Linear(3*3*32, 128)
        self.linear2_mu = nn.Linear(128, latent_dims) # Generate mus
        self.linear2_sigma = nn.Linear(128, latent_dims) # Generate sigmas

        self.N = torch.distributions.Normal(0,1)
        # TODO: Try make below not hardcoded to GPU
        self.N.loc = self.N.loc.cuda() # Make sampling on GPU
        self.N.scale = self.N.scale.cuda() # Make sampling on GPU
        self.kl = 0 
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.batch2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu = self.linear2_mu(x)
        sigma = torch.exp(self.linear2_sigma(x))
        z = mu + sigma * self.N.sample(mu.shape)
        # Calculate KL Divergence
        self.kl = (sigma**2+mu**2-torch.log(sigma)-1/2).mean()
        return z

class Decoder(nn.Module):
    # Can be used for Vanilla autoencoder or VAE 
    # since variational only affects encoder.
    def __init__(self, latent_dims):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dims, 128),
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