import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import wandb
import os
from uuid import uuid4
from pydantic import BaseModel
from typing import Any

# Configuration model using Pydantic
class VAEConfig(BaseModel):
    latent_dim: int = 20
    hidden_dim: int = 400
    input_dim: int = 784  # 28x28 MNIST images
    batch_size: int = 128
    epochs: int = 10
    learning_rate: float = 1e-3
    device: Any = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir: str = "checkpoints"
    n_images_to_log: int = 8

# Instantiate config
config = VAEConfig()

# Initialize wandb with config
wandb.init(project="vae_mnist", config=config.model_dump())

# Set random seed for reproducibility
torch.manual_seed(42)

# Define the VAE model
class VAE(nn.Module):
    def __init__(self, config: VAEConfig):
        super(VAE, self).__init__()
        
        # Encoder
        self.fc1 = nn.Linear(config.input_dim, config.hidden_dim)
        self.fc21 = nn.Linear(config.hidden_dim, config.latent_dim)  # Mean
        self.fc22 = nn.Linear(config.hidden_dim, config.latent_dim)  # Log variance
        
        # Decoder
        self.fc3 = nn.Linear(config.latent_dim, config.hidden_dim)
        self.fc4 = nn.Linear(config.hidden_dim, config.input_dim)
        
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))
    
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, config.input_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Loss function
def loss_function(recon_x, x, mu, logvar):
    # Ensure target is in [0, 1] by reversing normalization
    target = (x.view(-1, config.input_dim) + 1) / 2  # Convert [-1, 1] to [0, 1]
    BCE = F.binary_cross_entropy(recon_x, target, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts to [0, 1]
    transforms.Normalize((0.5,), (0.5,))  # Normalizes to [-1, 1]
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

# Initialize model, optimizer
model = VAE(config).to(config.device)
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

# Model checkpointing
best_loss = float('inf')
os.makedirs(config.checkpoint_dir, exist_ok=True)

# Training loop
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(config.device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')
    
    avg_train_loss = train_loss / len(train_loader.dataset)
    print(f'====> Epoch: {epoch} Average train loss: {avg_train_loss:.4f}')
    return avg_train_loss

# Test function with image logging
def test(epoch):
    global best_loss
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(config.device)
            recon, mu, logvar = model(data)
            test_loss += loss_function(recon, data, mu, logvar).item()
            
            # Log images for the first batch
            if i == 0:
                n_images = config.n_images_to_log
                originals = (data[:n_images].cpu().view(-1, 28, 28) + 1) / 2  # Convert [-1, 1] to [0, 1] for visualization
                reconstructions = recon[:n_images].cpu().view(-1, 28, 28)
                
                # Create comparison images
                fig, axes = plt.subplots(2, n_images, figsize=(n_images * 2, 4))
                for j in range(n_images):
                    axes[0, j].imshow(originals[j], cmap='gray')
                    axes[0, j].axis('off')
                    axes[0, j].set_title('Original')
                    axes[1, j].imshow(reconstructions[j], cmap='gray')
                    axes[1, j].axis('off')
                    axes[1, j].set_title('Reconstructed')
                
                wandb.log({
                    "epoch": epoch,
                    "original_vs_reconstructed": wandb.Image(fig)
                })
                plt.close(fig)
    
    avg_test_loss = test_loss / len(test_loader.dataset)
    print(f'====> Test set loss: {avg_test_loss:.4f}')
    
    # Save best model
    if avg_test_loss < best_loss:
        best_loss = avg_test_loss
        checkpoint_path = os.path.join(config.checkpoint_dir, f"vae_mnist_best_{str(uuid4())}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_loss
        }, checkpoint_path)
        print(f"Saved best model to {checkpoint_path}")
        wandb.save(checkpoint_path)
    
    return avg_test_loss

# Run training and log to wandb
for epoch in range(1, config.epochs + 1):
    train_loss = train(epoch)
    test_loss = test(epoch)
    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "test_loss": test_loss
    })

# Generate samples
def generate_samples(n_samples=16):
    model.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, config.latent_dim).to(config.device)
        samples = model.decode(z).cpu()
        
        # Plot and log generated samples
        fig = plt.figure(figsize=(4, 4))
        for i in range(n_samples):
            plt.subplot(4, 4, i + 1)
            plt.imshow(samples[i].reshape(28, 28), cmap='gray')
            plt.axis('off')
        wandb.log({"generated_samples": wandb.Image(fig)})
        plt.close(fig)

# Generate and log samples
generate_samples()
wandb.finish()