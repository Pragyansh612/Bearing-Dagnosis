import torch
import torch.nn as nn

class AttentionFusion(nn.Module):
    def __init__(self, feature_dim=256, num_branches=3, num_classes=4):
        super().__init__()
        self.feature_dim  = feature_dim
        self.num_branches = num_branches

        self.attention = nn.Sequential(
            nn.Linear(feature_dim * num_branches, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_branches),
            nn.Softmax(dim=-1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, *features):
        concat  = torch.cat(list(features), dim=-1)
        weights = self.attention(concat)
        stacked = torch.stack(list(features), dim=1)
        fused   = (stacked * weights.unsqueeze(-1)).sum(dim=1)
        logits  = self.classifier(fused)
        return logits, weights
