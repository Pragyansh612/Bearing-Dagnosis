import torch
import torch.nn as nn
import torch.nn.functional as F

class FreqChannelAttention(nn.Module):
    """Attention over frequency bins - learns which frequencies matter"""
    def __init__(self, n_freq_bins, reduction=8):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(n_freq_bins, n_freq_bins // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(n_freq_bins // reduction, n_freq_bins),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, F, T)
        freq_avg = x.mean(dim=[1, 3])  # (B, F)
        attn = self.attn(freq_avg)     # (B, F)
        return x * attn.unsqueeze(1).unsqueeze(3)

class FrequencyBranch(nn.Module):
    def __init__(self, in_shape=(64, 64), out_dim=256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # (32, 32, 32)

            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # (64, 16, 16)

            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # (128, 8, 8)
        )

        # Frequency attention applied after first pool
        self.freq_attn = FreqChannelAttention(n_freq_bins=32)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

        # Save last conv layer reference for Grad-CAM
        self.last_conv = self.features[-3]  # last Conv2d before final pool

    def forward(self, x):
        # x: (B, 64, 64)
        x = x.unsqueeze(1)  # (B, 1, 64, 64)

        # First block
        x = self.features[0](x)   # Conv
        x = self.features[1](x)   # BN
        x = self.features[2](x)   # ReLU
        x = self.features[3](x)   # MaxPool → (B, 32, 32, 32)

        # Frequency attention
        x = self.freq_attn(x)

        # Remaining blocks
        x = self.features[4](x)
        x = self.features[5](x)
        x = self.features[6](x)
        x = self.features[7](x)   # → (B, 64, 16, 16)

        x = self.features[8](x)
        x = self.features[9](x)
        x = self.features[10](x)
        x = self.features[11](x)  # → (B, 128, 8, 8)

        return self.head(x)
