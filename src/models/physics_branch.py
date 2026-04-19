import torch
import torch.nn as nn

class PhysicsBranch(nn.Module):
    """
    Physics + Metadata branch.
    Input: 30 physics features + 4 metadata features = 34 dimensions
    Metadata: fixedSpeed, assetType, mean_RPM, samplingRate
    """
    def __init__(self, input_dim=34, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)
