import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import wandb
import os
from sklearn.model_selection import train_test_split


class NoiseModel(nn.Module):
    """
    UNet to predict the noise given x_t, t, and class label y.
    """

    def __init__(self, time_dim=256, num_classes=10):
        super(NoiseModel, self).__init__()

        # Time embedding
        self.time_dim = time_dim
        self.time_embedding = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Class embedding
        self.class_embedding = nn.Embedding(num_classes, time_dim)

        # Initial projection (28x28 -> 28x28)
        self.initial_conv = nn.Conv2d(1, 64, 3, padding=1)

        # Encoder
        # 28x28 -> 14x14
        self.enc1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # 14x14 -> 7x7
        self.enc2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        # 7x7 -> 3x3
        self.enc3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        # Bottleneck (3x3 -> 3x3)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU()
        )

        # Decoder
        # 3x3 -> 7x7
        self.dec3 = nn.Sequential(
            nn.Conv2d(1024, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        # 7x7 -> 14x14
        self.dec2 = nn.Sequential(
            nn.Conv2d(512, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # 14x14 -> 28x28
        self.dec1 = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # Final layer (28x28 -> 28x28)
        self.final_conv = nn.Conv2d(64, 1, 3, padding=1)

        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2, ceil_mode=True)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # Time projection layers (for combined embedding)
        self.time_proj1 = nn.Conv2d(time_dim, 128, 1)
        self.time_proj2 = nn.Conv2d(time_dim, 256, 1)
        self.time_proj3 = nn.Conv2d(time_dim, 512, 1)

    def forward(self, x, t, y):
        # Time embedding
        t = t.unsqueeze(-1).float()
        t_emb = self.time_embedding(t)

        # Class embedding
        y_emb = self.class_embedding(y)

        # Combine embeddings
        emb = t_emb + y_emb
        emb = emb.view(-1, self.time_dim, 1, 1)

        # Initial convolution
        x0 = self.initial_conv(x)

        # Encoder path
        e1 = self.enc1(x0)
        e1_pooled = self.pool(e1)
        e2 = self.enc2(e1_pooled)
        e2_pooled = self.pool(e2)
        e3 = self.enc3(e2_pooled)
        e3_pooled = self.pool(e3)

        # Bottleneck
        b = self.bottleneck(e3_pooled)

        # Time embeddings
        t1 = self.time_proj1(emb)
        t2 = self.time_proj2(emb)
        t3 = self.time_proj3(emb)

        # Decoder path with proper size alignment
        up_b = self.up(b)
        e3_adjusted = F.interpolate(
            e3 + t3, size=(8, 8), mode="bilinear", align_corners=True
        )
        d3 = self.dec3(torch.cat([up_b, e3_adjusted], dim=1))

        up_d3 = self.up(d3)
        e2_adjusted = F.interpolate(
            e2 + t2, size=(16, 16), mode="bilinear", align_corners=True
        )
        d2 = self.dec2(torch.cat([up_d3, e2_adjusted], dim=1))

        up_d2 = self.up(d2)
        e1_adjusted = F.interpolate(
            e1 + t1, size=(32, 32), mode="bilinear", align_corners=True
        )
        d1 = self.dec1(torch.cat([up_d2, e1_adjusted], dim=1))

        # Final adjustment to 28x28
        d1_adjusted = F.interpolate(
            d1, size=(28, 28), mode="bilinear", align_corners=True
        )
        out = self.final_conv(d1_adjusted)

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
            -1, 1, 1, 1
        )
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(
            1.0 - self.alphas_cumprod.to(device)[t]
        ).view(-1, 1, 1, 1)

        return (
            sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise,
            noise,
        )


def train(
    noise_model: nn.Module,
    forward_process: ForwardProcess,
    device: torch.device,
    num_epochs: int = 10,
    batch_size: int = 128,
    model_save_path: str = "./best_model.pth",
):
    # Initialize wandb
    wandb.init(
        project="conditional-diffusion-mnist",
        config={
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "num_timesteps": forward_process.num_timesteps,
            "learning_rate": 1e-3,
        },
    )

    # Load MNIST dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )

    # Split into training and validation sets (80-20 split)
    indices = list(range(len(dataset)))
    train_indices, val_indices = train_test_split(
        indices, test_size=0.2, random_state=42
    )

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

    optimizer = torch.optim.Adam(noise_model.parameters(), lr=1e-3)
    best_val_loss = float("inf")

    noise_model.train()
    for epoch in range(num_epochs):
        # Training
        train_loss = 0.0
        for batch_idx, (x_0, labels) in enumerate(train_loader):
            x_0 = x_0.to(device)
            labels = labels.to(device)
            batch_size = x_0.shape[0]

            t = torch.randint(
                0, forward_process.num_timesteps, (batch_size,), device=device
            )
            x_t, noise = forward_process.q_sample(device, x_0, t)
            predicted_noise = noise_model(x_t, t, labels)
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

        # Validation
        noise_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_0, labels in val_loader:
                x_0 = x_0.to(device)
                labels = labels.to(device)
                batch_size = x_0.shape[0]

                t = torch.randint(
                    0, forward_process.num_timesteps, (batch_size,), device=device
                )
                x_t, noise = forward_process.q_sample(device, x_0, t)
                predicted_noise = noise_model(x_t, t, labels)
                loss = F.mse_loss(predicted_noise, noise)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        wandb.log({"epoch": epoch, "val_loss": avg_val_loss})
        print(
            f"Epoch {epoch}, Avg Train Loss: {avg_train_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}"
        )

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(noise_model.state_dict(), model_save_path)
            print(
                f"Saved best model at epoch {epoch} with val loss: {best_val_loss:.4f}"
            )

        # Generate samples with labels
        y_sample = torch.randint(0, 10, (16,), device=device)
        samples = sample(noise_model, forward_process, device, n_samples=16, y=y_sample)
        samples = (samples + 1) / 2  # Convert to [0,1]
        samples = samples.cpu().numpy()
        y_sample = y_sample.cpu().numpy()

        # Create figure with labels
        n_samples = samples.shape[0]
        grid_size = int(np.ceil(np.sqrt(n_samples)))
        fig, axes = plt.subplots(
            grid_size, grid_size, figsize=(grid_size * 2, grid_size * 2)
        )
        fig.suptitle(f"Generated Samples at Epoch {epoch}", fontsize=16)
        plt.subplots_adjust(wspace=0.1, hspace=0.3)  # Adjust hspace for label space

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

        # Save to file
        samples_grid = torchvision.utils.make_grid(
            torch.from_numpy(samples), nrow=4, normalize=True
        )
        torchvision.utils.save_image(samples_grid, f"generated_mnist_epoch_{epoch}.png")

        # Log to wandb
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
    noise_model: NoiseModel, diffusion: ForwardProcess, device, n_samples=16, y=None
):
    if y is None:
        raise ValueError(
            "Class labels 'y' must be provided for conditional generation."
        )
    if y.shape[0] != n_samples:
        raise ValueError("y must have shape (n_samples,)")

    noise_model.eval()
    x = torch.randn(n_samples, 1, 28, 28).to(device)
    y = y.to(device)

    for t in reversed(range(diffusion.num_timesteps)):
        t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)
        predicted_noise = noise_model(x, t_tensor, y)

        alpha = diffusion.alphas[t]
        alpha_cumprod = diffusion.alphas_cumprod[t]
        beta = diffusion.betas[t]

        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)

        x = (1 / torch.sqrt(alpha)) * (
            x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * predicted_noise
        ) + torch.sqrt(beta) * noise

    return x


def visualize_samples(samples, title="Generated MNIST Samples", labels=None):
    """
    Visualize generated samples in a grid, optionally with labels
    """
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
def visualize_denoising_process(model, diffusion, device, n_samples=4, y=None):
    if y is None:
        y = torch.randint(0, 10, (n_samples,), device=device)

    x = torch.randn(n_samples, 1, 28, 28).to(device)
    y = y.to(device)
    intermediates = []

    for t in reversed(range(0, diffusion.num_timesteps, 100)):
        t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)
        predicted_noise = model(x, t_tensor, y)

        alpha = diffusion.alphas[t]
        alpha_cumprod = diffusion.alphas_cumprod[t]
        beta = diffusion.betas[t]

        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)

        x = (1 / torch.sqrt(alpha)) * (
            x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * predicted_noise
        ) + torch.sqrt(beta) * noise

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

    # Initialize models
    noise_model = NoiseModel(num_classes=10).to(device)
    forward_process = ForwardProcess()

    # Train with model saving and wandb logging
    train(
        noise_model,
        forward_process,
        device,
        num_epochs=100,
        model_save_path="best_model.pth",
    )

    # Generate and visualize samples for a specific digit (e.g., digit 7)
    y_gen = torch.full((16,), 7, device=device)
    samples = sample(noise_model, forward_process, device, n_samples=16, y=y_gen)
    samples = (samples + 1) / 2
    visualize_samples(samples, title="Generated Digit 7", labels=y_gen.cpu().numpy())

    # Log final samples to wandb
    samples_grid = torchvision.utils.make_grid(samples, nrow=4, normalize=True)
    wandb.log(
        {"final_samples": wandb.Image(samples_grid, caption="Final Generated Digit 7")}
    )
    wandb.finish()
