import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from noise_model import NoiseModel


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


# 3. Training Setup
def train(
    noise_model: nn.Module,
    forward_process: ForwardProcess,
    device: torch.device,
    num_epochs: int = 10,
    batch_size: int = 128,
):
    # Load MNIST dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(noise_model.parameters(), lr=1e-3)

    noise_model.train()
    for epoch in range(num_epochs):
        for batch_idx, (x_0, _) in enumerate(dataloader):
            x_0 = x_0.to(device)
            batch_size = x_0.shape[0]

            # Sample random timesteps
            t = torch.randint(
                0, forward_process.num_timesteps, (batch_size,), device=device
            )

            # Add noise to images
            x_t, noise = forward_process.q_sample(device, x_0, t)

            # Predict noise
            predicted_noise = noise_model(x_t, t)

            # Compute loss
            loss = F.mse_loss(predicted_noise, noise)

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        # Generate samples
        samples = sample(noise_model, forward_process, device)
        # Convert samples to proper range [-1, 1] -> [0, 1]
        samples = (samples + 1) / 2

        # visualize_denoising_process(model, diffusion, device)

        # Optionally save the image
        samples_grid = torchvision.utils.make_grid(samples, nrow=4, normalize=True)
        torchvision.utils.save_image(samples_grid, f"generated_mnist_epoch_{epoch}.png")


# 4. Sampling Function
@torch.no_grad()
def sample(noise_model: NoiseModel, diffusion: ForwardProcess, device, n_samples=16):
    noise_model.eval()
    x = torch.randn(n_samples, 1, 28, 28).to(device)

    for t in reversed(range(diffusion.num_timesteps)):
        t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)
        predicted_noise = noise_model(x, t_tensor)

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


# 5. Visualization Function
def visualize_samples(samples, title="Generated MNIST Samples"):
    """
    Visualize generated samples in a grid
    """
    # Convert to CPU and numpy
    samples = samples.cpu().numpy()
    n_samples = samples.shape[0]

    # Calculate grid size
    grid_size = int(np.ceil(np.sqrt(n_samples)))

    # Create figure
    fig, axes = plt.subplots(
        grid_size, grid_size, figsize=(grid_size * 2, grid_size * 2)
    )
    fig.suptitle(title, fontsize=16)

    # Remove gaps between subplots
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    # Plot each sample
    for i in range(n_samples):
        row = i // grid_size
        col = i % grid_size

        # Remove batch and channel dimensions, normalize to [0,1]
        img = samples[i, 0]

        axes[row, col].imshow(img, cmap="gray")
        axes[row, col].axis("off")

    # Remove empty subplots
    for i in range(n_samples, grid_size * grid_size):
        row = i // grid_size
        col = i % grid_size
        axes[row, col].axis("off")

    plt.show()


# To visualize the denoising process:
@torch.no_grad()
def visualize_denoising_process(model, diffusion, device, n_samples=4):
    x = torch.randn(n_samples, 1, 28, 28).to(device)
    intermediates = []

    for t in reversed(range(0, diffusion.num_timesteps, 100)):  # Sample every 100 steps
        t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)
        predicted_noise = model(x, t_tensor)

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

    # Visualize all intermediates
    for i, intermediate in enumerate(intermediates):
        intermediate = (intermediate + 1) / 2
        visualize_samples(intermediate, f"Timestep {diffusion.num_timesteps - i*100}")


# 5. Main execution
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    # Initialize models
    noise_model = NoiseModel().to(device)
    forward_process = ForwardProcess()

    # Train the model
    train(noise_model, forward_process, device, num_epochs=100)
