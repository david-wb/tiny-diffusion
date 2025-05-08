import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt


class NoiseModel(nn.Module):
    """
    UNet to predict the noise given x_t and t.
    """

    def __init__(self, time_dim=256):
        super(NoiseModel, self).__init__()

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
        self.pool = nn.MaxPool2d(2, ceil_mode=True)  # Use ceil_mode to handle odd sizes
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

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
        e1 = self.enc1(x0)  # (B,128,28,28)
        e1_pooled = self.pool(e1)  # (B,128,14,14)
        e2 = self.enc2(e1_pooled)  # (B,256,14,14)
        e2_pooled = self.pool(e2)  # (B,256,7,7)
        e3 = self.enc3(e2_pooled)  # (B,512,7,7)
        e3_pooled = self.pool(e3)  # (B,512,4,4)

        # Bottleneck
        b = self.bottleneck(e3_pooled)  # (B,512,4,4)

        # Time embeddings
        t1 = self.time_proj1(t_emb)  # (B,128,1,1)
        t2 = self.time_proj2(t_emb)  # (B,256,1,1)
        t3 = self.time_proj3(t_emb)  # (B,512,1,1)

        # Decoder path with proper size alignment
        up_b = self.up(b)  # (B,512,8,8)
        # Crop or pad e3 (7x7) to match up_b (8x8)
        e3_adjusted = F.interpolate(
            e3 + t3, size=(8, 8), mode="bilinear", align_corners=True
        )
        d3 = self.dec3(torch.cat([up_b, e3_adjusted], dim=1))  # (B,1024,8,8)

        up_d3 = self.up(d3)  # (B,256,16,16)
        # Crop or pad e2 (14x14) to match up_d3 (16x16)
        e2_adjusted = F.interpolate(
            e2 + t2, size=(16, 16), mode="bilinear", align_corners=True
        )
        d2 = self.dec2(torch.cat([up_d3, e2_adjusted], dim=1))  # (B,512,16,16)

        up_d2 = self.up(d2)  # (B,128,32,32)
        # Crop e1 (28x28) to match up_d2 after adjustment
        e1_adjusted = F.interpolate(
            e1 + t1, size=(32, 32), mode="bilinear", align_corners=True
        )
        d1 = self.dec1(torch.cat([up_d2, e1_adjusted], dim=1))  # (B,256,32,32)

        # Final adjustment to 28x28
        d1_adjusted = F.interpolate(
            d1, size=(28, 28), mode="bilinear", align_corners=True
        )
        out = self.final_conv(d1_adjusted)  # (B,1,28,28)

        return out
