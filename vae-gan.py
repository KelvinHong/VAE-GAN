import torchvision
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from utils import *

EPOCHS = 50

train_dataset = torchvision.datasets.FashionMNIST(
    root = "./",
    train = True, 
    download = True,
    transform = transforms.ToTensor()
)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
i =  next(iter(train_dataloader))[0]
print(i.shape)

