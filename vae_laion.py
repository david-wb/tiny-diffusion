import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import wandb
import os
from uuid import uuid4
from pydantic import BaseModel
from typing import Any
from datasets import load_dataset
from PIL import Image
import numpy as np
import requests
from io import BytesIO


# Configuration model using Pydantic
class VAEConfig(BaseModel):
    latent_dim: int = 128
    hidden_channels: int = 64
    input_channels: int = 3  # RGB images
    image_size: int = 256  # LAION images (resized to 256x256)
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 1e-4
    device: Any = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir: str = "checkpoints"
    n_images_to_log: int = 8


# Instantiate config
config = VAEConfig()

# Set random seed for reproducibility
torch.manual_seed(42)


# Define the Convolutional VAE model
class VAE(nn.Module):
    def __init__(self, config: VAEConfig):
        super(VAE, self).__init__()
        self.config = config

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(
                config.input_channels, 32, 4, stride=2, padding=1
            ),  # [batch, 32, 128, 128]
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # [batch, 64, 64, 64]
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # [batch, 128, 32, 32]
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # [batch, 256, 16, 16]
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 16 * 16, config.latent_dim),  # mu
            nn.Linear(256 * 16 * 16, config.latent_dim),  # logvar
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, 256 * 16 * 16),
            nn.ReLU(),
            nn.Unflatten(1, (256, 16, 16)),
            nn.ConvTranspose2d(
                256, 128, 4, stride=2, padding=1
            ),  # [batch, 128, 32, 32]
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # [batch, 64, 64, 64]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # [batch, 32, 128, 128]
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, config.input_channels, 4, stride=2, padding=1
            ),  # [batch, 3, 256, 256]
            nn.Sigmoid(),  # Output in [0, 1]
        )

    def encode(self, x):
        h = self.encoder[:-2](x)
        mu = self.encoder[-2](h)
        logvar = self.encoder[-1](h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# Loss function
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


# Load LAION dataset
def load_laion_dataset():
    # Load dataset without streaming, limiting to a subset for memory
    train = load_dataset("laion/laion2B-en-aesthetic", split="train[:10000]")
    return train


# Define transforms for LAION images
transform = transforms.Compose(
    [
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),  # Converts to [0, 1]
    ]
)


# Custom dataset wrapper for LAION
class LAIONDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        try:
            # Assume the dataset has a 'URL' key for image URLs
            url = sample["URL"]
            response = requests.get(url, timeout=5)
            response.raise_for_status()  # Raise an error for bad responses
            image = Image.open(BytesIO(response.content)).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image
        except (requests.RequestException, Image.UnidentifiedImageError) as e:
            print(f"Skipping sample {idx} due to error: {e}")
            # Return a zero tensor of correct shape to avoid breaking DataLoader
            return torch.zeros((3, config.image_size, config.image_size))


# Initialize dataset and dataloader
laion_train = load_laion_dataset()
train_dataset = LAIONDataset(laion_train, transform)
train_loader = DataLoader(
    train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0, pin_memory=True
)


# Initialize model, optimizer
model = VAE(config).to(config.device)
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)


# Model checkpointing
best_loss = float("inf")
os.makedirs(config.checkpoint_dir, exist_ok=True)


# Training loop
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(config.device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_dataset)}] "
                f"({100. * batch_idx / len(train_loader):.0f}%)\tLoss: {loss.item() / len(data):.6f}"
            )

    avg_train_loss = train_loss / len(train_dataset)
    print(f"====> Epoch: {epoch} Average train loss: {avg_train_loss:.4f}")
    return avg_train_loss


# Test function with image logging
def test(epoch):
    global best_loss
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(train_loader):
            data = data.to(config.device)
            recon, mu, logvar = model(data)
            test_loss += loss_function(recon, data, mu, logvar).item()

            if i == 0:
                n_images = config.n_images_to_log
                originals = (
                    data[:n_images].cpu().permute(0, 2, 3, 1)
                )  # [batch, 3, 256, 256] -> [batch, 256, 256, 3]
                reconstructions = recon[:n_images].cpu().permute(0, 2, 3, 1)

                fig, axes = plt.subplots(2, n_images, figsize=(n_images * 2, 4))
                for j in range(n_images):
                    axes[0, j].imshow(originals[j].numpy())
                    axes[0, j].axis("off")
                    axes[0, j].set_title("Original")
                    axes[1, j].imshow(reconstructions[j].numpy())
                    axes[1, j].axis("off")
                    axes[1, j].set_title("Reconstructed")

                wandb.log(
                    {"epoch": epoch, "original_vs_reconstructed": wandb.Image(fig)}
                )
                plt.close(fig)

    avg_test_loss = test_loss / len(train_dataset)
    print(f"====> Test set loss: {avg_test_loss:.4f}")

    if avg_test_loss < best_loss:
        best_loss = avg_test_loss
        checkpoint_path = os.path.join(config.checkpoint_dir, f"vae_laion_best.pth")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
                "config": config.model_dump(),
            },
            checkpoint_path,
        )
        print(f"Saved best model to {checkpoint_path}")
        wandb.save(checkpoint_path)

    return avg_test_loss


if __name__ == "__main__":
    # Initialize wandb with config
    wandb.init(project="vae_laion", config=config.model_dump())

    # Run training and log to wandb
    for epoch in range(1, config.epochs + 1):
        train_loss = train(epoch)
        test_loss = test(epoch)
        wandb.log({"epoch": epoch, "train_loss": train_loss, "test_loss": test_loss})

    # Generate samples
    def generate_samples(n_samples=16):
        model.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, config.latent_dim).to(config.device)
            samples = model.decode(z).cpu().permute(0, 2, 3, 1)

            fig = plt.figure(figsize=(4, 4))
            for i in range(n_samples):
                plt.subplot(4, 4, i + 1)
                plt.imshow(samples[i].numpy())
                plt.axis("off")
            wandb.log({"generated_samples": wandb.Image(fig)})
            plt.close(fig)

    # Generate and log samples
    generate_samples()
    wandb.finish()