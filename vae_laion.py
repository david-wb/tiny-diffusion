import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import vgg16, VGG16_Weights
import matplotlib.pyplot as plt
import wandb
import os
import json
from pydantic import BaseModel
from typing import Any
from datasets import load_dataset
from PIL import Image
import numpy as np
import requests
from io import BytesIO
import hashlib
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# Configuration model using Pydantic
class VAEConfig(BaseModel):
    latent_dim: int = 128
    hidden_channels: int = 64
    input_channels: int = 3  # RGB images
    image_size: int = 256  # LAION images (resized to 256x256)
    batch_size: int = 4
    epochs: int = 100
    learning_rate: float = 1e-4
    device: Any = torch.device("cuda")
    checkpoint_dir: str = "checkpoints"
    image_cache_dir: str = "data/laion"  # Directory for caching images
    failed_urls_cache: str = "data/failed_urls.json"  # File for caching failed URLs
    n_images_to_log: int = 8
    log_interval: int = 10
    beta: float = 1.0  # Added for β-VAE


# Instantiate config
config = VAEConfig()

# Set random seed for reproducibility
torch.manual_seed(42)


# Self-Attention Layer
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch, C, H, W = x.size()
        query = self.query(x).view(batch, -1, H * W).permute(0, 2, 1)
        key = self.key(x).view(batch, -1, H * W)
        attention = F.softmax(torch.bmm(query, key), dim=-1)
        value = self.value(x).view(batch, -1, H * W)
        out = torch.bmm(value, attention.permute(0, 2, 1)).view(batch, C, H, W)
        return self.gamma * out + x


# Residual Block with Spectral Normalization
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.utils.spectral_norm(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.utils.spectral_norm(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        )
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return x + residual


# VAE with Single Latent Variable
class VAE(nn.Module):
    def __init__(self, config: VAEConfig):
        super(VAE, self).__init__()
        self.config = config

        # Encoder
        self.encoder = nn.ModuleList(
            [
                nn.Sequential(
                    nn.utils.spectral_norm(
                        nn.Conv2d(config.input_channels, 32, 4, stride=2, padding=1)
                    ),  # [batch, 32, 128, 128]
                    nn.ReLU(),
                    ResidualBlock(32),
                    SelfAttention(32),
                ),
                nn.Sequential(
                    nn.utils.spectral_norm(
                        nn.Conv2d(32, 64, 4, stride=2, padding=1)
                    ),  # [batch, 64, 64, 64]
                    nn.ReLU(),
                    ResidualBlock(64),
                    SelfAttention(64),
                ),
                nn.Sequential(
                    nn.utils.spectral_norm(
                        nn.Conv2d(64, 128, 4, stride=2, padding=1)
                    ),  # [batch, 128, 32, 32]
                    nn.ReLU(),
                    ResidualBlock(128),
                ),
                nn.Sequential(
                    nn.utils.spectral_norm(
                        nn.Conv2d(128, 256, 4, stride=2, padding=1)
                    ),  # [batch, 256, 16, 16]
                    nn.ReLU(),
                    ResidualBlock(256),
                ),
            ]
        )
        self.fc_mu = nn.Linear(256 * 16 * 16, config.latent_dim)
        self.fc_logvar = nn.Linear(256 * 16 * 16, config.latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(config.latent_dim, 256 * 16 * 16)
        self.decoder = nn.ModuleList(
            [
                nn.Sequential(
                    nn.utils.spectral_norm(
                        nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
                    ),  # [batch, 128, 32, 32]
                    nn.ReLU(),
                    ResidualBlock(128),
                    SelfAttention(128),
                ),
                nn.Sequential(
                    nn.utils.spectral_norm(
                        nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
                    ),  # [batch, 64, 64, 64]
                    nn.ReLU(),
                    ResidualBlock(64),
                    SelfAttention(64),
                ),
                nn.Sequential(
                    nn.utils.spectral_norm(
                        nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
                    ),  # [batch, 32, 128, 128]
                    nn.ReLU(),
                    ResidualBlock(32),
                ),
                nn.Sequential(
                    nn.utils.spectral_norm(
                        nn.ConvTranspose2d(
                            32, config.input_channels, 4, stride=2, padding=1
                        )
                    ),  # [batch, 3, 256, 256]
                    nn.Sigmoid(),  # Output in [0, 1]
                ),
            ]
        )

        # Perceptual loss network (pretrained VGG16)
        vgg = (
            vgg16(weights=VGG16_Weights.DEFAULT).features[:16].eval().to(config.device)
        )
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg

    def encode(self, x):
        h = x
        for layer in self.encoder:
            h = layer(h)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        h = h.view(h.size(0), 256, 16, 16)
        for layer in self.decoder:
            h = layer(h)
        return h

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    def perceptual_loss(self, recon_x, x):
        recon_features = self.vgg(recon_x)
        target_features = self.vgg(x)
        return F.mse_loss(recon_features, target_features, reduction="sum")

    def loss_function(self, recon_x, x, mu, logvar):
        # Reconstruction loss (BCE + Perceptual)
        bce_loss = F.binary_cross_entropy(recon_x, x, reduction="sum")
        perc_loss = self.perceptual_loss(recon_x, x)
        recon_loss = bce_loss + 0.1 * perc_loss  # Weight perceptual loss

        # KL divergence for single latent
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Log individual loss components for debugging
        if (
            torch.any(torch.isnan(bce_loss))
            or torch.any(torch.isnan(perc_loss))
            or torch.any(torch.isnan(kld))
        ):
            print(
                f"Loss components: BCE={bce_loss.item()}, Perceptual={perc_loss.item()}, KLD={kld.item()}"
            )

        # Total loss with beta weighting
        return recon_loss + self.config.beta * kld


# Custom dataset wrapper for LAION with local caching
class LAIONDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
        # Create image cache directory
        os.makedirs(config.image_cache_dir, exist_ok=True)
        # Initialize failed URLs cache
        self.failed_urls = set()
        os.makedirs(os.path.dirname(config.failed_urls_cache), exist_ok=True)
        if os.path.exists(config.failed_urls_cache):
            try:
                with open(config.failed_urls_cache, "r") as f:
                    self.failed_urls = set(json.load(f))
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading failed URLs cache: {e}")
                self.failed_urls = set()

    def save_failed_urls(self):
        """Save the failed URLs to a JSON file."""
        try:
            with open(config.failed_urls_cache, "w") as f:
                json.dump(list(self.failed_urls), f)
        except IOError as e:
            print(f"Error saving failed URLs cache: {e}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        try:
            url = sample["URL"]
            if url in self.failed_urls:
                # print(f"Skipping previously failed URL: {url}")
                return torch.zeros((3, config.image_size, config.image_size))

            url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()
            cache_path = os.path.join(config.image_cache_dir, f"{url_hash}.jpg")

            if os.path.exists(cache_path):
                try:
                    image = Image.open(cache_path).convert("RGB")
                except (OSError, Image.UnidentifiedImageError) as e:
                    print(f"Corrupted cache file {cache_path}, redownloading: {e}")
                    os.remove(cache_path)
                else:
                    if self.transform:
                        image = self.transform(image)
                    return image

            # Retry mechanism for downloading
            session = requests.Session()
            retries = Retry(
                total=1, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504]
            )
            session.mount("http://", HTTPAdapter(max_retries=retries))
            session.mount("https://", HTTPAdapter(max_retries=retries))
            response = session.get(url, timeout=5)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")

            image.save(cache_path, "JPEG")

            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            # print(f"Skipping sample {idx} due to error: {e}")
            self.failed_urls.add(url)
            self.save_failed_urls()
            return torch.zeros((3, config.image_size, config.image_size))


# Load LAION dataset with download mode
def load_laion_dataset():
    train = load_dataset("laion/laion2B-en-aesthetic", split="train[:10000]")
    return train


# Define transforms for LAION images
transform = transforms.Compose(
    [
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),  # Converts to [0, 1]
    ]
)

# Initialize dataset and dataloader
laion_train = load_laion_dataset()
train_dataset = LAIONDataset(laion_train, transform)
train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
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
        if torch.all(data == 0):
            print(
                f"Batch {batch_idx} contains all-zero images, likely due to failed downloads."
            )
            continue  # Skip batches with all-zero images
        data = data.to(config.device)
        optimizer.zero_grad()
        recon, mu, logvar = model(data)
        loss = model.loss_function(recon, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % config.log_interval == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_dataset)}] "
                f"({100. * batch_idx / len(train_loader):.0f}%)\tLoss: {loss.item() / len(data):.6f}"
            )
            n_images = min(
                config.n_images_to_log, data.size(0)
            )  # Use min of config and batch size
            originals = (
                data[:n_images].detach().cpu().permute(0, 2, 3, 1)
            )  # [batch, 3, 256, 256] -> [batch, 256, 256, 3]
            reconstructions = recon[:n_images].detach().cpu().permute(0, 2, 3, 1)

            if n_images == 1:
                fig, axes = plt.subplots(2, 1, figsize=(2, 4))
                axes = axes.reshape(2, 1)  # Ensure 2D array for consistency
            else:
                fig, axes = plt.subplots(2, n_images, figsize=(n_images * 2, 4))
            for j in range(n_images):
                axes[0, j].imshow(originals[j].numpy())
                axes[0, j].axis("off")
                axes[0, j].set_title("Original")
                axes[1, j].imshow(reconstructions[j].numpy())
                axes[1, j].axis("off")
                axes[1, j].set_title("Reconstructed")
            wandb.log(
                {
                    "epoch": epoch,
                    "step": batch_idx,
                    "original_vs_reconstructed": wandb.Image(fig),
                    "batch_train_loss": loss.item() / len(data),
                }
            )
            plt.close(fig)

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
            if torch.all(data == 0):
                print(
                    f"Test Batch {i} contains all-zero images, likely due to failed downloads."
                )
                continue  # Skip batches with all-zero images
            data = data.to(config.device)
            recon, mu, logvar = model(data)
            test_loss += model.loss_function(recon, data, mu, logvar).item()

            if i % config.log_interval == 0:
                n_images = min(
                    config.n_images_to_log, data.size(0)
                )  # Use min of config and batch size
                originals = (
                    data[:n_images].cpu().permute(0, 2, 3, 1)
                )  # [batch, 3, 256, 256] -> [batch, 256, 256, 3]
                reconstructions = recon[:n_images].cpu().permute(0, 2, 3, 1)

                if n_images == 1:
                    fig, axes = plt.subplots(2, 1, figsize=(2, 4))
                    axes = axes.reshape(2, 1)  # Ensure 2D array for consistency
                else:
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


# Main execution
if __name__ == "__main__":
    torch.cuda.empty_cache()
    # Initialize wandb with config
    wandb.init(project="vae_laion", config=config.model_dump())

    # Run training and log to wandb
    for epoch in range(1, config.epochs + 1):
        train_loss = train(epoch)
        test_loss = test(epoch)
        wandb.log({"epoch": epoch, "train_loss": train_loss, "test_loss": test_loss})

    # Generate and log samples
    generate_samples()
    wandb.finish()
