import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import wandb
from sklearn.model_selection import train_test_split
import os
from dataclasses import dataclass
from vae import VAE, VAEConfig


class NoiseModel(nn.Module):
    def __init__(self, time_dim=256, num_classes=10, latent_dim=20):
        super(NoiseModel, self).__init__()

        self.time_dim = time_dim
        self.latent_dim = latent_dim

        # Time embedding
        self.time_embedding = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Class embedding
        self.class_embedding = nn.Embedding(num_classes, time_dim)

        # Initial projection
        self.initial_fc = nn.Linear(latent_dim, 512)

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        self.enc2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.enc3 = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        # Decoder
        self.dec3 = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.dec2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        self.dec1 = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )

        # Final layer
        self.final_fc = nn.Linear(512, latent_dim)

        # Time projection layers
        self.time_proj1 = nn.Linear(time_dim, 64)
        self.time_proj2 = nn.Linear(time_dim, 128)
        self.time_proj3 = nn.Linear(time_dim, 256)

    def forward(self, x, t, y):
        t = t.unsqueeze(-1).float()
        t_emb = self.time_embedding(t)
        y_emb = self.class_embedding(y)
        emb = t_emb + y_emb

        x0 = self.initial_fc(x)
        e1 = self.enc1(x0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.bottleneck(e3)

        t1 = self.time_proj1(emb)
        t2 = self.time_proj2(emb)
        t3 = self.time_proj3(emb)

        d3 = self.dec3(torch.cat([b + t1, e3], dim=1))
        d2 = self.dec2(torch.cat([d3 + t2, e2], dim=1))
        d1 = self.dec1(torch.cat([d2 + t3, e1], dim=1))
        out = self.final_fc(d1)

        return out


class ForwardProcess:
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, device, x_0, t):
        noise = torch.randn_like(x_0).to(device)
        sqrt_alphas_cumprod_t = torch.sqrt(self.alphas_cumprod.to(device)[t]).view(
            -1, 1
        )
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(
            1.0 - self.alphas_cumprod.to(device)[t]
        ).view(-1, 1)
        return (
            sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise,
            noise,
        )


def train(
    vae: VAE,
    config: VAEConfig,
    noise_model: nn.Module,
    forward_process: ForwardProcess,
    device: torch.device,
    num_epochs: int = 10,
    batch_size: int = 128,
    model_save_path: str = "./best_model.pth",
):
    wandb.init(
        project="conditional-latent-diffusion-mnist",
        config={
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "num_timesteps": forward_process.num_timesteps,
            "learning_rate": 1e-3,
        },
    )

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )

    train_indices, val_indices = train_test_split(
        list(range(len(dataset))), test_size=0.2, random_state=42
    )

    train_sampler = SubsetRandomSampler(train_indices)
    val_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=SubsetRandomSampler(val_indices)
    )
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)

    optimizer = torch.optim.Adam(noise_model.parameters(), lr=1e-3)
    vae.eval()  # VAE is pre-trained
    best_val_loss = float("inf")

    noise_model.train()
    for epoch in range(num_epochs):
        train_loss = 0.0
        for batch_idx, (x_0, labels) in enumerate(train_loader):
            x_0 = x_0.to(device)
            labels = labels.to(device)
            batch_size = x_0.shape[0]

            with torch.no_grad():
                mu, logvar = vae.encode(x_0.view(-1, config.input_dim))
                z_0 = vae.reparameterize(mu, logvar)

            t = torch.randint(
                0, forward_process.num_timesteps, (batch_size,), device=device
            )
            z_t, noise = forward_process.q_sample(device, z_0, t)
            predicted_noise = noise_model(z_t, t, labels)
            loss = F.mse_loss(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if batch_idx % 100 == 0:
                print(
                    f"Epoch {epoch}, Batch {batch_idx}, Train Loss: {loss.item():.4f}"
                )

        avg_train_loss = train_loss / len(train_loader)
        wandb.log({"epoch": epoch, "train_loss": avg_train_loss})

        noise_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_0, labels in val_loader:
                x_0 = x_0.to(device)
                labels = labels.to(device)
                batch_size = x_0.shape[0]

                mu, logvar = vae.encode(x_0.view(-1, config.input_dim))
                z_0 = vae.reparameterize(mu, logvar)
                t = torch.randint(
                    0, forward_process.num_timesteps, (batch_size,), device=device
                )
                z_t, noise = forward_process.q_sample(device, z_0, t)
                predicted_noise = noise_model(z_t, t, labels)
                loss = F.mse_loss(predicted_noise, noise)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        wandb.log({"epoch": epoch, "val_loss": avg_val_loss})
        print(
            f"Epoch {epoch}, Avg Train Loss: {avg_train_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(noise_model.state_dict(), model_save_path)
            print(
                f"Saved best model at epoch {epoch} with val loss: {best_val_loss:.4f}"
            )

        y_sample = torch.randint(0, 10, (16,), device=device)
        samples = sample(
            vae, noise_model, forward_process, device, n_samples=16, y=y_sample
        )
        samples = (samples + 1) / 2
        samples = samples.cpu().numpy()
        y_sample = y_sample.cpu().numpy()

        n_samples = samples.shape[0]
        grid_size = int(np.ceil(np.sqrt(n_samples)))
        fig, axes = plt.subplots(
            grid_size, grid_size, figsize=(grid_size * 2, grid_size * 2)
        )
        fig.suptitle(f"Generated Samples at Epoch {epoch}", fontsize=16)
        plt.subplots_adjust(wspace=0.1, hspace=0.3)

        for i in range(n_samples):
            row = i // grid_size
            col = i % grid_size
            img = samples[i, 0]
            axes[row, col].imshow(img, cmap="gray")
            axes[row, col].set_title(f"Label: {y_sample[i]}", fontsize=10)
            axes[row, col].axis("off")

        for i in range(n_samples, grid_size * grid_size):
            row = i // grid_size
            col = i % grid_size
            axes[row, col].axis("off")

        samples_grid = torchvision.utils.make_grid(
            torch.from_numpy(samples), nrow=4, normalize=True
        )
        torchvision.utils.save_image(samples_grid, f"generated_mnist_epoch_{epoch}.png")
        wandb.log(
            {
                "epoch": epoch,
                "samples": wandb.Image(
                    fig, caption=f"Generated samples at epoch {epoch}"
                ),
            }
        )
        plt.close(fig)

        noise_model.train()


@torch.no_grad()
def sample(
    vae: VAE,
    noise_model: NoiseModel,
    diffusion: ForwardProcess,
    device,
    n_samples=16,
    y=None,
):
    if y is None:
        raise ValueError(
            "Class labels 'y' must be provided for conditional generation."
        )
    if y.shape[0] != n_samples:
        raise ValueError("y must have shape (n_samples,)")

    vae.eval()
    noise_model.eval()
    z = torch.randn(n_samples, vae.config.latent_dim).to(device)
    y = y.to(device)

    for t in reversed(range(diffusion.num_timesteps)):
        t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)
        predicted_noise = noise_model(z, t_tensor, y)

        alpha = diffusion.alphas[t]
        alpha_cumprod = diffusion.alphas_cumprod[t]
        beta = diffusion.betas[t]

        if t > 0:
            noise = torch.randn_like(z)
        else:
            noise = torch.zeros_like(z)

        z = (1 / torch.sqrt(alpha)) * (
            z - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * predicted_noise
        ) + torch.sqrt(beta) * noise

    x = vae.decode(z).view(-1, 1, 28, 28)
    return x


def visualize_samples(samples, title="Generated MNIST Samples", labels=None):
    samples = samples.cpu().numpy()
    n_samples = samples.shape[0]
    grid_size = int(np.ceil(np.sqrt(n_samples)))

    fig, axes = plt.subplots(
        grid_size, grid_size, figsize=(grid_size * 2, grid_size * 2)
    )
    fig.suptitle(title, fontsize=16)
    plt.subplots_adjust(wspace=0.1, hspace=0.3)

    for i in range(n_samples):
        row = i // grid_size
        col = i % grid_size
        img = samples[i, 0]
        axes[row, col].imshow(img, cmap="gray")
        if labels is not None:
            axes[row, col].set_title(f"Label: {labels[i]}", fontsize=10)
        axes[row, col].axis("off")

    for i in range(n_samples, grid_size * grid_size):
        row = i // grid_size
        col = i % grid_size
        axes[row, col].axis("off")

    plt.show()


@torch.no_grad()
def visualize_denoising_process(vae, model, diffusion, device, n_samples=4, y=None):
    if y is None:
        y = torch.randint(0, 10, (n_samples,), device=device)

    vae.eval()
    model.eval()
    z = torch.randn(n_samples, vae.config.latent_dim).to(device)
    y = y.to(device)
    intermediates = []

    for t in reversed(range(0, diffusion.num_timesteps, 100)):
        t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)
        predicted_noise = model(z, t_tensor, y)

        alpha = diffusion.alphas[t]
        alpha_cumprod = diffusion.alphas_cumprod[t]
        beta = diffusion.betas[t]

        if t > 0:
            noise = torch.randn_like(z)
        else:
            noise = torch.zeros_like(z)

        z = (1 / torch.sqrt(alpha)) * (
            z - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * predicted_noise
        ) + torch.sqrt(beta) * noise

        x = vae.decode(z).view(-1, 1, 28, 28)
        intermediates.append(x.clone())

    for i, intermediate in enumerate(intermediates):
        intermediate = (intermediate + 1) / 2
        visualize_samples(
            intermediate,
            f"Timestep {diffusion.num_timesteps - i*100}",
            labels=y.cpu().numpy(),
        )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    checkpoint_path = "checkpoints/vae_mnist_best.pth"
    if os.path.exists(checkpoint_path):
        print(f"Loading VAE from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config_dict = checkpoint.get("config", {})
        config = VAEConfig(**config_dict)
        vae = VAE(config).to(device)
        vae.load_state_dict(checkpoint["model_state_dict"])
        print("VAE and config loaded successfully")
    else:
        print(f"Checkpoint not found at {checkpoint_path}, using default config")
        config = VAEConfig()
        vae = VAE(config).to(device)

    noise_model = NoiseModel(num_classes=10, latent_dim=config.latent_dim).to(device)
    forward_process = ForwardProcess()

    train(
        vae,
        config,
        noise_model,
        forward_process,
        device,
        num_epochs=100,
        model_save_path="best_latent_model.pth",
    )

    y_gen = torch.full((16,), 7, device=device)
    samples = sample(vae, noise_model, forward_process, device, n_samples=16, y=y_gen)
    samples = (samples + 1) / 2
    visualize_samples(samples, title="Generated Digit 7", labels=y_gen.cpu().numpy())

    samples_grid = torchvision.utils.make_grid(samples, nrow=4, normalize=True)
    wandb.log(
        {"final_samples": wandb.Image(samples_grid, caption="Final Generated Digit 7")}
    )
    wandb.finish()
