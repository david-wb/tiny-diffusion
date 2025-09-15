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
from datasets import load_dataset
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from PIL import Image
import hashlib
import json
from io import BytesIO
from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
import types
from torch.amp import GradScaler, autocast  # Updated AMP imports
from tqdm import tqdm
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("precache.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Enable CUDA optimizations
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# Configuration
config = types.SimpleNamespace(
    image_cache_dir="./data/laion_image_cache",
    failed_urls_cache="./data/failed_urls.json",
    image_size=256,
)


def samples_to_wandb_images(samples, prompts):
    for sample, prompt in zip(samples, prompts):
        yield wandb.Image(sample, caption=prompt)


# Custom dataset wrapper for LAION with local caching
class LAIONDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
        os.makedirs(config.image_cache_dir, exist_ok=True)
        self.failed_urls = set()
        os.makedirs(os.path.dirname(config.failed_urls_cache), exist_ok=True)
        if os.path.exists(config.failed_urls_cache):
            try:
                with open(config.failed_urls_cache, "r") as f:
                    self.failed_urls = set(json.load(f))
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error loading failed URLs cache: {e}")
                self.failed_urls = set()

    def save_failed_urls(self):
        try:
            with open(config.failed_urls_cache, "w") as f:
                json.dump(list(self.failed_urls), f)
        except IOError as e:
            logger.error(f"Error saving failed URLs cache: {e}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        url = sample["URL"]
        text = sample["TEXT"]
        try:
            if url in self.failed_urls:
                # logger.warning(f"Skipping cached failed URL: {url}")
                return torch.zeros((3, config.image_size, config.image_size)), ""

            url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()
            cache_path = os.path.join(config.image_cache_dir, f"{url_hash}.jpg")

            if os.path.exists(cache_path):
                try:
                    image = Image.open(cache_path).convert("RGB")
                except (OSError, Image.UnidentifiedImageError) as e:
                    logger.error(
                        f"Corrupted cache file {cache_path}, redownloading: {e}"
                    )
                    os.remove(cache_path)
                    image = None
                else:
                    if image is not None:
                        if self.transform:
                            image_tensor = self.transform(image)
                        return image_tensor, text

            session = requests.Session()
            retries = Retry(
                total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504]
            )
            session.mount("http://", HTTPAdapter(max_retries=retries))
            session.mount("https://", HTTPAdapter(max_retries=retries))
            response = session.get(url, timeout=15)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")

            image.save(cache_path, "JPEG", quality=95)

            if self.transform:
                image_tensor = self.transform(image)
            return image_tensor, text
        except Exception as e:
            logger.error(f"Error processing URL {url} at index {idx}: {e}")
            self.failed_urls.add(url)
            self.save_failed_urls()
            return torch.zeros((3, config.image_size, config.image_size)), ""


def load_laion_dataset():
    train = load_dataset("laion/laion2B-en-aesthetic", split="train[:10000]")
    return train


def check_disk_space(path, required_space):
    """Check if there is enough disk space for caching."""
    total, used, free = shutil.disk_usage(path)
    free_gb = free / (1024**3)
    required_gb = required_space / (1024**3)
    if free_gb < required_gb:
        logger.error(
            f"Insufficient disk space: {free_gb:.2f} GB available, {required_gb:.2f} GB required"
        )
        raise RuntimeError(f"Need at least {required_gb:.2f} GB free disk space")
    logger.info(
        f"Disk space check: {free_gb:.2f} GB available, {required_gb:.2f} GB required"
    )


def precache_dataset(dataset, max_samples=None):
    logger.info("Starting dataset pre-caching...")
    max_samples = max_samples or len(dataset)
    max_samples = min(max_samples, len(dataset))
    required_space = max_samples * 250 * 1024  # ~250 KB per image
    check_disk_space(config.image_cache_dir, required_space)

    def cache_sample(idx):
        try:
            dataset[idx]
            return idx, True
        except Exception as e:
            logger.error(f"Failed to cache sample {idx}: {e}")
            return idx, False

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(cache_sample, idx): idx for idx in range(max_samples)
        }
        for future in tqdm(
            as_completed(futures), total=max_samples, desc="Pre-caching"
        ):
            idx, success = future.result()
            if not success:
                logger.warning(f"Sample {idx} failed to cache")

    logger.info("Pre-caching complete.")


def get_text_embeds(tokenizer, text_model, device, texts):
    text_model.to(device)
    inputs = tokenizer(
        texts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(
        device
    )  # Ensure inputs are on device
    with torch.no_grad():
        outputs = text_model(inputs.input_ids)
        text_embeds = outputs.last_hidden_state[:, -1, :].to(
            device
        )  # Ensure embeds stay on device
    return text_embeds


class NoiseModel(nn.Module):
    def __init__(self, time_dim=768):
        super(NoiseModel, self).__init__()
        self.time_dim = time_dim
        self.time_embedding = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.initial_conv = nn.Conv2d(4, 32, 3, padding=1)
        self.enc1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.dec3 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(384, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(192, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.final_conv = nn.Conv2d(64, 4, 3, padding=1)
        self.time_proj1 = nn.Conv2d(time_dim, 64, 1)
        self.time_proj2 = nn.Conv2d(time_dim, 128, 1)
        self.time_proj3 = nn.Conv2d(time_dim, 256, 1)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, x, t, text_embeds):
        t = t.unsqueeze(-1).float()
        t_emb = self.time_embedding(t)
        emb = t_emb + text_embeds
        emb = emb.view(-1, self.time_dim, 1, 1)
        x0 = self.initial_conv(x)
        e1 = self.enc1(x0)
        e1_pooled = self.pool(e1)
        e2 = self.enc2(e1_pooled)
        e2_pooled = self.pool(e2)
        e3 = self.enc3(e2_pooled)
        e3_pooled = self.pool(e3)
        b = self.bottleneck(e3_pooled)
        t1 = self.time_proj1(emb)
        t2 = self.time_proj2(emb)
        t3 = self.time_proj3(emb)
        up_b = self.up(b)
        e3_adjusted = e3 + t3
        d3 = self.dec3(torch.cat([up_b, e3_adjusted], dim=1))
        up_d3 = self.up(d3)
        e2_adjusted = e2 + t2
        d2 = self.dec2(torch.cat([up_d3, e2_adjusted], dim=1))
        up_d2 = self.up(d2)
        e1_adjusted = e1 + t1
        d1 = self.dec1(torch.cat([up_d2, e1_adjusted], dim=1))
        out = self.final_conv(d1)
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
    vae,
    tokenizer,
    text_model,
    scaling_factor,
    num_epochs: int = 10,
    batch_size: int = 8,
    model_save_path: str = "./best_model.pth",
):
    wandb.init(
        project="laion-diffusion-model",
        config={
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "num_timesteps": forward_process.num_timesteps,
            "learning_rate": 1e-4,
            "image_size": 256,
            "latent_size": 32,
        },
    )
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    hf_dataset = load_laion_dataset()
    dataset = LAIONDataset(hf_dataset, transform)
    precache_dataset(dataset, max_samples=50000)
    indices = list(range(len(dataset)))
    train_indices, val_indices = train_test_split(
        indices, test_size=0.2, random_state=42
    )
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
    )
    optimizer = torch.optim.Adam(noise_model.parameters(), lr=1e-4)
    scaler = GradScaler("cuda")  # Updated API
    best_val_loss = float("inf")

    def get_text_embeds_local(texts):
        return get_text_embeds(tokenizer, text_model, device, texts)

    noise_model.train()
    prompts_for_sample = [
        "a photo of a cat",
        "a photo of a dog",
        "a photo of a horse",
        "a photo of a cow",
    ]
    text_embeds_for_sample = get_text_embeds_local(prompts_for_sample)
    for epoch in range(num_epochs):
        train_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            images, texts = batch
            images = images.to(device)
            text_embeds = get_text_embeds_local(texts)
            batch_size_actual = images.shape[0]
            with autocast("cuda"):  # Updated API
                vae.to(device)
                latents = vae.encode(images).latent_dist.sample()
                latents = latents * scaling_factor
                x_0 = latents
                t = torch.randint(
                    0,
                    forward_process.num_timesteps,
                    (batch_size_actual,),
                    device=device,
                )
                x_t, noise = forward_process.q_sample(device, x_0, t)
                # Log device of tensors for debugging
                logger.debug(
                    f"x_t device: {x_t.device}, t device: {t.device}, text_embeds device: {text_embeds.device}"
                )
                predicted_noise = noise_model(x_t, t, text_embeds)
                loss = F.mse_loss(predicted_noise, noise)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            if batch_idx % 10 == 0:
                wandb.log(
                    {"epoch": epoch, "batch": batch_idx, "batch_train_loss": loss}
                )
            if batch_idx % 100 == 0:
                with torch.no_grad():
                    samples = sample(
                        noise_model,
                        forward_process,
                        device,
                        text_embeds=text_embeds_for_sample,
                        vae=vae,
                        scaling_factor=scaling_factor,
                    )
                    wandb_images = list(
                        samples_to_wandb_images(samples, prompts_for_sample)
                    )
                    wandb.log(
                        {
                            "epoch": epoch,
                            "batch": batch_idx,
                            "sampled_images": wandb_images,
                        }
                    )
        avg_train_loss = train_loss / len(train_loader)
        wandb.log({"epoch": epoch, "epoch_train_loss": avg_train_loss})
        noise_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images, texts = batch
                images = images.to(device)
                text_embeds = get_text_embeds_local(texts)
                batch_size_actual = images.shape[0]
                with autocast("cuda"):  # Updated API
                    vae.to(device)
                    latents = vae.encode(images).latent_dist.sample()
                    latents = latents * scaling_factor
                    x_0 = latents
                    t = torch.randint(
                        0,
                        forward_process.num_timesteps,
                        (batch_size_actual,),
                        device=device,
                    )
                    x_t, noise = forward_process.q_sample(device, x_0, t)
                    predicted_noise = noise_model(x_t, t, text_embeds)
                    loss = F.mse_loss(predicted_noise, noise)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        wandb.log({"epoch": epoch, "val_loss": avg_val_loss})
        logger.info(
            f"Epoch {epoch}, Avg Train Loss: {avg_train_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}"
        )
        torch.save(noise_model.state_dict(), f"model_epoch_{epoch}.pth")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(noise_model.state_dict(), model_save_path)
            logger.info(
                f"Saved best model at epoch {epoch} with val loss: {best_val_loss:.4f}"
            )
        samples = sample(
            noise_model,
            forward_process,
            device,
            text_embeds=text_embeds_for_sample,
            vae=vae,
            scaling_factor=scaling_factor,
        )
        samples_grid = torchvision.utils.make_grid(samples, nrow=4)
        fig, axes = plt.subplots(2, 2, figsize=(16, 16))
        fig.suptitle(f"Generated Samples at Epoch {epoch}", fontsize=16)
        plt.subplots_adjust(wspace=0.1, hspace=0.3)
        for i, ax in enumerate(axes.flat):
            if i < 16:
                img = np.transpose(samples[i].cpu().numpy(), (1, 2, 0))
                ax.imshow(img)
                ax.set_title(prompts_for_sample[i], fontsize=8)
                ax.axis("off")
        torchvision.utils.save_image(samples_grid, f"generated_epoch_{epoch}.png")
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
    wandb.finish()


@torch.no_grad()
def sample(
    noise_model: NoiseModel,
    diffusion: ForwardProcess,
    device,
    text_embeds=None,
    vae=None,
    scaling_factor=1.0,
):
    if text_embeds is None:
        raise ValueError("Text embeddings must be provided for conditional generation.")

    if text_embeds is not None:
        n_samples = text_embeds.shape[0]
    else:
        n_samples = 1

    noise_model.eval()
    noise_model = torch.compile(noise_model)
    x = torch.randn(n_samples, 4, 32, 32).to(device)
    for t in reversed(range(diffusion.num_timesteps)):
        t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)
        with autocast("cuda"):  # Updated API
            predicted_noise = noise_model(x, t_tensor, text_embeds)
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
    with torch.no_grad():
        with autocast("cuda"):  # Updated API
            vae.to(device)
            decoded = vae.decode(x / scaling_factor).sample
            images = (decoded / 2 + 0.5).clamp(0, 1)
            # Check for NaN/Inf in images
            if torch.isnan(images).any() or torch.isinf(images).any():
                logger.error("NaN or Inf detected in images tensor")
                # Replace NaN/Inf with 0
                images = torch.where(
                    torch.logical_or(torch.isnan(images), torch.isinf(images)),
                    torch.zeros_like(images),
                    images,
                )
            # Ensure images is float32 for Matplotlib compatibility
            images = images.to(torch.float32)
    return images


def visualize_samples(samples, title="Generated Samples", prompts=None):
    samples = samples.cpu().numpy()
    n_samples = samples.shape[0]
    grid_size = int(np.ceil(np.sqrt(n_samples)))
    fig, axes = plt.subplots(
        grid_size, grid_size, figsize=(grid_size * 3, grid_size * 3)
    )
    if grid_size == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=16)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    for i in range(n_samples):
        row = i // grid_size
        col = i % grid_size
        img = np.transpose(samples[i], (1, 2, 0))
        axes[row][col].imshow(img)
        if prompts is not None:
            axes[row][col].set_title(
                prompts[i][:15] + "..." if len(prompts[i]) > 15 else prompts[i],
                fontsize=9,
            )
        axes[row][col].axis("off")
    for i in range(n_samples, grid_size * grid_size):
        row = i // grid_size
        col = i % grid_size
        axes[row][col].axis("off")
    plt.show()


@torch.no_grad()
def visualize_denoising_process(
    model,
    diffusion,
    device,
    n_samples=4,
    prompts=None,
    vae=None,
    scaling_factor=1.0,
    tokenizer=None,
    text_model=None,
):
    if prompts is None:
        prompts = ["a photo of a cat"] * n_samples
    text_embeds = get_text_embeds(tokenizer, text_model, device, prompts)
    model = torch.compile(model)
    x = torch.randn(n_samples, 4, 32, 32, device=device)
    intermediates = []
    for t in reversed(range(diffusion.num_timesteps)):
        t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)
        with autocast("cuda"):  # Updated API
            predicted_noise = model(x, t_tensor, text_embeds)
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
        if t % 100 == 0 or t == 0:
            with torch.no_grad():
                with autocast("cuda"):  # Updated API
                    vae.to(device)
                    decoded = vae.decode(x / scaling_factor).sample
                    img = (decoded / 2 + 0.5).clamp(0, 1)
                    intermediates.append(img.clone())
    for i, intermediate in enumerate(intermediates):
        step = diffusion.num_timesteps - i * 100
        visualize_samples(intermediate, f"Timestep {step}", prompts)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")
    vae = AutoencoderKL.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="vae"
    ).to(device)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(
        device
    )
    scaling_factor = vae.config.scaling_factor
    noise_model = NoiseModel(time_dim=768).to(device)
    # Ensure all model parameters and buffers are on device
    for name, param in noise_model.named_parameters():
        if param.device != device:
            logger.warning(f"Parameter {name} is on {param.device}, moving to {device}")
            param.data = param.data.to(device)
    for name, buffer in noise_model.named_buffers():
        if buffer.device != device:
            logger.warning(f"Buffer {name} is on {buffer.device}, moving to {device}")
            buffer.data = buffer.data.to(device)
    noise_model = torch.compile(noise_model)
    forward_process = ForwardProcess()
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    dataset = LAIONDataset(load_laion_dataset(), transform)
    precache_dataset(dataset, max_samples=50000)
    train(
        noise_model,
        forward_process,
        device,
        vae=vae,
        tokenizer=tokenizer,
        text_model=text_model,
        scaling_factor=scaling_factor,
        num_epochs=10,
        batch_size=8,
        model_save_path="best_model.pth",
    )
    prompts_gen = [
        "a photo of a cat",
        "a photo of a dog",
        "a photo of a horse",
        "a photo of a cow",
        "a photo of a bird",
        "a photo of a fish",
        "a photo of a tree",
        "a photo of a flower",
        "a photo of a house",
        "a photo of a car",
        "a photo of a person",
        "a photo of a mountain",
        "a photo of a river",
        "a photo of a sky",
        "a photo of a sun",
        "a photo of a moon",
    ]
    text_embeds = get_text_embeds(tokenizer, text_model, device, prompts_gen)
    samples = sample(
        noise_model,
        forward_process,
        device,
        text_embeds=text_embeds,
        vae=vae,
        scaling_factor=scaling_factor,
    )
    samples_grid = torchvision.utils.make_grid(samples, nrow=4)
    wandb.log(
        {"final_samples": wandb.Image(samples_grid, caption="Final Generated Samples")}
    )
    wandb.finish()
