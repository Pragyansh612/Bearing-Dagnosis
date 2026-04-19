import torch
import torch.nn as nn

class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.skip = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm1d(out_ch),
        ) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.skip(x))

class TimeDomainBranch(nn.Module):
    """Multi-scale 1D ResNet: processes both short (1024) and long (2048) windows"""
    def __init__(self, out_dim=256):
        super().__init__()

        # Short scale branch (first 1024 samples)
        self.short_entry = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True)
        )
        self.short_blocks = nn.Sequential(
            ResBlock1D(32, 64),
            ResBlock1D(64, 128),
        )
        self.short_pool = nn.AdaptiveAvgPool1d(1)

        # Long scale branch (all 2048 samples)
        self.long_entry = nn.Sequential(
            nn.Conv1d(1, 32, 15, padding=7, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True)
        )
        self.long_blocks = nn.Sequential(
            ResBlock1D(32, 64),
            ResBlock1D(64, 128),
        )
        self.long_pool = nn.AdaptiveAvgPool1d(1)

        # Fuse both scales
        self.fuse = nn.Sequential(
            nn.Linear(256, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        # x: (B, 2048)
        x = x.unsqueeze(1)  # (B, 1, 2048)

        # Short scale: first 1024 samples
        x_short = x[:, :, :1024]
        x_short = self.short_pool(self.short_blocks(self.short_entry(x_short))).squeeze(-1)

        # Long scale: all 2048 samples
        x_long = self.long_pool(self.long_blocks(self.long_entry(x))).squeeze(-1)

        # Concatenate and fuse: 128 + 128 = 256
        x_cat = torch.cat([x_short, x_long], dim=-1)
        return self.fuse(x_cat)
