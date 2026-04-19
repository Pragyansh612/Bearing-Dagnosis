import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BearingAutoencoder(nn.Module):
    """
    Autoencoder trained ONLY on healthy data fused features.
    Learns what healthy looks like in the 256-dim feature space.
    At inference: high reconstruction error = anomaly/unknown fault.
    """
    def __init__(self, input_dim=256, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Linear(64, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(128, input_dim),
        )

        self.threshold      = None
        self.mean_error     = None
        self.std_error      = None

    def forward(self, x):
        z    = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

    def reconstruction_error(self, x):
        """Per-sample MSE reconstruction error"""
        recon, _ = self.forward(x)
        return F.mse_loss(recon, x, reduction='none').mean(dim=-1)

    def compute_threshold(self, healthy_features, n_sigma=3.0):
        """
        Compute threshold from healthy training data.
        Threshold = mean + n_sigma * std of reconstruction errors.
        """
        self.eval()
        errors = []
        with torch.no_grad():
            for i in range(0, len(healthy_features), 256):
                batch = healthy_features[i:i+256]
                if isinstance(batch, np.ndarray):
                    batch = torch.from_numpy(batch).float()
                    if next(self.parameters()).is_cuda:
                        batch = batch.cuda()
                err = self.reconstruction_error(batch)
                errors.extend(err.cpu().numpy().tolist())

        errors = np.array(errors)
        self.mean_error = float(errors.mean())
        self.std_error  = float(errors.std())
        self.threshold  = self.mean_error + n_sigma * self.std_error

        print(f"  Autoencoder threshold computed:")
        print(f"    Mean error:  {self.mean_error:.6f}")
        print(f"    Std error:   {self.std_error:.6f}")
        print(f"    Threshold:   {self.threshold:.6f} ({n_sigma}σ)")
        return self.threshold

    def normalized_error(self, x):
        """
        Returns normalized reconstruction error.
        > 3.0 means more than 3 sigma above healthy baseline.
        """
        raw_err = self.reconstruction_error(x)
        if self.std_error and self.std_error > 0:
            return (raw_err - self.mean_error) / self.std_error
        return raw_err

    def is_anomaly(self, x, threshold_sigma=3.0):
        norm_err = self.normalized_error(x)
        return norm_err > threshold_sigma, norm_err
