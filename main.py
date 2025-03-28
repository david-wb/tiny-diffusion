import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

class UNet(nn.Module):
    def __init__(self, time_dim=256):
        super(UNet, self).__init__()
        
        # Time embedding
        self.time_dim = time_dim
        self.time_embedding = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        
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
            nn.ReLU()
        )
        
        # 14x14 -> 7x7
        self.enc2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        # 7x7 -> 3x3
        self.enc3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        
        # Bottleneck (3x3 -> 3x3)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        
        # Decoder
        # 3x3 -> 7x7
        self.dec3 = nn.Sequential(
            nn.Conv2d(1024, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        # 7x7 -> 14x14
        self.dec2 = nn.Sequential(
            nn.Conv2d(512, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # 14x14 -> 28x28
        self.dec1 = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Final layer (28x28 -> 28x28)
        self.final_conv = nn.Conv2d(64, 1, 3, padding=1)
        
        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2, ceil_mode=True)  # Use ceil_mode to handle odd sizes
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Time projection layers
        self.time_proj1 = nn.Conv2d(time_dim, 128, 1)
        self.time_proj2 = nn.Conv2d(time_dim, 256, 1)
        self.time_proj3 = nn.Conv2d(time_dim, 512, 1)

    def forward(self, x, t):
        # Time embedding
        t = t.unsqueeze(-1).float()
        t_emb = self.time_embedding(t)
        t_emb = t_emb.view(-1, self.time_dim, 1, 1)
        
        # Initial convolution (B,1,28,28) -> (B,64,28,28)
        x0 = self.initial_conv(x)
        
        # Encoder path
        e1 = self.enc1(x0)          # (B,128,28,28)
        e1_pooled = self.pool(e1)   # (B,128,14,14)
        e2 = self.enc2(e1_pooled)   # (B,256,14,14)
        e2_pooled = self.pool(e2)   # (B,256,7,7)
        e3 = self.enc3(e2_pooled)   # (B,512,7,7)
        e3_pooled = self.pool(e3)   # (B,512,4,4)
        
        # Bottleneck
        b = self.bottleneck(e3_pooled)  # (B,512,4,4)
        
        # Time embeddings
        t1 = self.time_proj1(t_emb)  # (B,128,1,1)
        t2 = self.time_proj2(t_emb)  # (B,256,1,1)
        t3 = self.time_proj3(t_emb)  # (B,512,1,1)
        
        # Decoder path with proper size alignment
        up_b = self.up(b)           # (B,512,8,8)
        # Crop or pad e3 (7x7) to match up_b (8x8)
        e3_adjusted = F.interpolate(e3 + t3, size=(8, 8), mode='bilinear', align_corners=True)
        d3 = self.dec3(torch.cat([up_b, e3_adjusted], dim=1))  # (B,1024,8,8)
        
        up_d3 = self.up(d3)        # (B,256,16,16)
        # Crop or pad e2 (14x14) to match up_d3 (16x16)
        e2_adjusted = F.interpolate(e2 + t2, size=(16, 16), mode='bilinear', align_corners=True)
        d2 = self.dec2(torch.cat([up_d3, e2_adjusted], dim=1))  # (B,512,16,16)
        
        up_d2 = self.up(d2)        # (B,128,32,32)
        # Crop e1 (28x28) to match up_d2 after adjustment
        e1_adjusted = F.interpolate(e1 + t1, size=(32, 32), mode='bilinear', align_corners=True)
        d1 = self.dec1(torch.cat([up_d2, e1_adjusted], dim=1))  # (B,256,32,32)
        
        # Final adjustment to 28x28
        d1_adjusted = F.interpolate(d1, size=(28, 28), mode='bilinear', align_corners=True)
        out = self.final_conv(d1_adjusted)  # (B,1,28,28)
        
        return out

# 2. Diffusion Process
class DiffusionModel:
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
    def q_sample(self, device, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0).to(device)
        
        sqrt_alphas_cumprod_t = torch.sqrt(self.alphas_cumprod.to(device)[t]).view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1. - self.alphas_cumprod.to(device)[t]).view(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

    def forward(self, x):
        return x

# 3. Training Setup
def train(model, diffusion, device, num_epochs=10, batch_size=128):
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    dataset = torchvision.datasets.MNIST(root='./data', train=True,
                                       download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, (x_0, _) in enumerate(dataloader):
            x_0 = x_0.to(device)
            batch_size = x_0.shape[0]
            
            # Sample random timesteps
            t = torch.randint(0, diffusion.num_timesteps, (batch_size,), device=device)
            
            # Add noise to images
            noise = torch.randn_like(x_0)
            x_t = diffusion.q_sample(device, x_0, t, noise)
            
            # Predict noise
            predicted_noise = model(x_t, t)
            
            # Compute loss
            loss = F.mse_loss(predicted_noise, noise)
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')

        # Generate samples
        samples = sample(model, diffusion, device)
        # Convert samples to proper range [-1, 1] -> [0, 1]
        samples = (samples + 1) / 2

        # visualize_denoising_process(model, diffusion, device)
        
        # Optionally save the image
        samples_grid = torchvision.utils.make_grid(samples, nrow=4, normalize=True)
        torchvision.utils.save_image(samples_grid, f'generated_mnist_epoch_{epoch}.png')

# 4. Sampling Function
@torch.no_grad()
def sample(model, diffusion, device, n_samples=16):
    model.eval()
    x = torch.randn(n_samples, 1, 28, 28).to(device)
    
    for t in reversed(range(diffusion.num_timesteps)):
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
    fig, axes = plt.subplots(grid_size, grid_size, 
                            figsize=(grid_size*2, grid_size*2))
    fig.suptitle(title, fontsize=16)
    
    # Remove gaps between subplots
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    
    # Plot each sample
    for i in range(n_samples):
        row = i // grid_size
        col = i % grid_size
        
        # Remove batch and channel dimensions, normalize to [0,1]
        img = samples[i, 0]
        
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].axis('off')
    
    # Remove empty subplots
    for i in range(n_samples, grid_size * grid_size):
        row = i // grid_size
        col = i % grid_size
        axes[row, col].axis('off')
    
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
    model = UNet().to(device)
    diffusion = DiffusionModel()
    
    # Train the model
    train(model, diffusion, device, num_epochs=100)
    
