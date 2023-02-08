"""
Variational AutoEncoder Implementation. 
Most code and pipeline referring to here: 
https://towardsdatascience.com/language-modeling-with-lstms-in-pytorch-381a26badcbf 

"""
import torchvision
from torchvision import transforms
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import VAE
from tqdm import tqdm
import os
import argparse
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import numpy as np

# Change configuration here
EPOCHS = 50
LATENT_DIM = 5
LR = 0.005
# Config ends

MODEL_ROOT = "./models/"
os.makedirs(MODEL_ROOT, exist_ok=True)

def train(model, dataloader, optimizer, criterion, device, epoch=None):
    epoch_rec_loss = 0
    epoch_latent_loss = 0
    model.train()
    num_batches = len(dataloader)
    it = iter(dataloader)
    desc = "Training: " if epoch is None else f"Training {epoch}: "
    for idx in tqdm(range(0, num_batches-1), 
            desc=desc):
        input = next(it)[0]
        optimizer.zero_grad()
        input = input.to(device)
        prediction = model(input)

        rec_loss = ((input - prediction) ** 2).sum() / input.shape[0]
        latent_loss = model.vencoder.kl

        loss = rec_loss + latent_loss
        loss.backward()
        optimizer.step()
        epoch_rec_loss += rec_loss.item()
        epoch_latent_loss += latent_loss.item()
    return epoch_rec_loss / num_batches, epoch_latent_loss /  num_batches

def evaluate(model, dataloader, criterion, device, epoch=None):
    # Unsupervised evaluation: Compare images to images instead of labels.
    epoch_rec_loss = 0
    epoch_latent_loss = 0
    model.eval()
    num_batches = len(dataloader)
    it = iter(dataloader)
    desc = "Evaluate: " if epoch is None else f"Evaluate {epoch}: "
    with torch.no_grad():
        for idx in tqdm(range(0, num_batches-1), 
                desc=desc):
            input = next(it)[0]
            input = input.to(device)
            prediction = model(input)
            
            rec_loss = ((input - prediction) ** 2).sum() / input.shape[0]
            latent_loss = model.vencoder.kl

            epoch_rec_loss += rec_loss.item()
            epoch_latent_loss += latent_loss.item()
    return epoch_rec_loss / num_batches, epoch_latent_loss /  num_batches

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog = 'Variatioanl AutoEncoder Handler',
                    description = """This program train or inference on 
                            a Variational AutoEncoder on FashionMNIST Dataset.""")
    parser.add_argument('--train', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-m', '--model-path', type=str, help="""
            During training, model will be saved on this path as MODEL_ROOT/[path], 
                if not provided, the model will be saved as MODEL_ROOT/best_vae.pth.
            During inference, model weights will be loaded from this path as MODEL_ROOT/[path], 
                can't be None. 
        """)
    parser.add_argument('-s', '--seed', type=int, help="Torch Seed for result reproducibility.")
    args = parser.parse_args()
    if (not args.train) and args.model_path is None:
        raise ValueError("When inferencing, provide a path to a trained model using the -m flag.")
    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        
    print(f"PyTorch Training using device: {device}.")
    train_dataset = torchvision.datasets.FashionMNIST(
        root = "./",
        train = True, 
        download = True,
        transform = transforms.ToTensor()
    )

    valid_dataset = torchvision.datasets.FashionMNIST(
        root = "./",
        train = False, 
        download = True,
        transform = transforms.ToTensor()
    )

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

    if args.train:
        # Train Mode
        # TODO: Implement LR Scheduling
        print("Train Mode")
        model = VAE(LATENT_DIM).to(device)
        model_name = "best_vae.pth" if args.model_path is None else args.model_path
        model_path = os.path.join(MODEL_ROOT, model_name)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.MSELoss()

        # Save model based on best validation score so far.
        best_valid_loss = float("inf")
        train_rec_losses = []
        train_lat_losses = []
        valid_rec_losses = []
        valid_lat_losses = []
        for epoch in range(1, EPOCHS+1):
            train_rec_loss, train_lat_loss = train(model, train_dataloader, 
                            optimizer, criterion, device, epoch)
            valid_rec_loss, valid_lat_loss = evaluate(model, valid_dataloader, 
                            criterion, device, epoch)
            print(f"Train Loss: Reconstruction {train_rec_loss:.6f} vs Latent {train_lat_loss:.6f}")
            print(f"Valid Loss: Reconstruction {valid_rec_loss:.6f} vs Latent {valid_lat_loss:.6f}")
            train_rec_losses.append(train_rec_loss)
            train_lat_losses.append(train_lat_loss)
            valid_rec_losses.append(valid_rec_loss)
            valid_lat_losses.append(valid_lat_loss)
            plt.plot(list(range(1, epoch+1)), train_rec_losses, label = "Train Rec Loss", color="red")
            plt.plot(list(range(1, epoch+1)), valid_rec_losses, label = "Valid Rec Loss", color="green")
            plt.legend()
            plt.title("VAE Training Progress - Reconstruction Losses")
            plt.savefig(os.path.join(MODEL_ROOT, os.path.splitext(model_name)[0] + "_rec.png" ))
            plt.clf()
            plt.plot(list(range(1, epoch+1)), train_lat_losses, label = "Train Lat Loss", color="orange")
            plt.plot(list(range(1, epoch+1)), valid_lat_losses, label = "Valid Lat Loss", color="blue")
            plt.legend()
            plt.title("VAE Training Progress - Latent Losses")
            plt.savefig(os.path.join(MODEL_ROOT, os.path.splitext(model_name)[0] + "_lat.png" ))
            plt.clf()
            valid_loss = valid_rec_loss + valid_lat_loss
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), model_path)
    
    else:
        # Evaluation Mode
        # Randomly choose 10 images from valid_dataloader and compare with their reconstruction.
        print("Evaluation Mode")
        model = VAE(LATENT_DIM).to(device)
        model_path = os.path.join(MODEL_ROOT, args.model_path)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        valid_take = 10
        n = len(valid_dataset)
        cands_ind = torch.randint(0, n, (valid_take,))
        imgs = torch.stack([valid_dataset[cand_ind][0] for cand_ind in cands_ind]).to(device)
        outputs = model(imgs)
        
        collage = torchvision.utils.make_grid(
            torch.cat([imgs, outputs]), nrow=5
        )
        collage = collage.cpu()
        plt.imshow(collage.permute(1, 2, 0)  )
        plt.title("VAE Reconstruction; Top Original, Bottom Reconstructed.")
        plt.show()

        # Inspect latent distribution by targets using all valid data
        label_to_latent = {
            i: [] for i in range(10)
        }
        for img, target in tqdm(valid_dataset, desc = "Inferencing"):
            z = model.vencoder(img.to(device).unsqueeze(dim=0))
            latent = z[0].tolist()
            label_to_latent[target].append(latent)
        # Calculate PCA
        pca = PCA(n_components=3)
        all_latents = np.concatenate([np.array(label_to_latent[i]) for i in range(10)], axis=0)
        all_labels = [i for i in range(10) for j in range(1000)]
        pca.fit(all_latents)
        # print(pca.explained_variance_ratio_) # Variance doesn't concentrate on the first 3 dimensions
        # Calculate Kmeans
        kmeans = KMeans(n_clusters=10)
        kmeans.fit(all_latents)
        print("Confusion matrix of truth and clustered labels on latents:")
        print("The more sparse, the better. It is better to be a permutation of a diagonal matrix.")
        print(confusion_matrix(all_labels, kmeans.labels_))
