import torch
from torch.utils.data import Dataset
import numpy as np
from src.preprocessing.signal_processor import (
    normalize_signal, compute_stft, compute_physics_features
)
from src.preprocessing.augmentation import augment_signal

class BearingDataset(Dataset):
    def __init__(self, signals, labels, rpm_arr, fs_arr, fault_freqs,
                 augment=False, stft_shape=(64, 64)):
        self.signals = signals
        self.labels  = labels
        self.rpm_arr = rpm_arr
        self.fs_arr  = fs_arr
        self.fault_freqs = fault_freqs
        self.augment = augment
        self.stft_shape = stft_shape

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx].copy()
        rpm    = float(self.rpm_arr[idx])
        fs     = float(self.fs_arr[idx])
        ff     = self.fault_freqs[idx]
        label  = int(self.labels[idx])

        # Augment
        if self.augment:
            signal, rpm = augment_signal(signal, rpm, fs)

        # Normalize
        signal = normalize_signal(signal)

        # Time domain input (2048,)
        x_time = torch.from_numpy(signal.copy()).float()

        # Frequency domain input: STFT spectrogram (64, 64)
        stft = compute_stft(signal, fs,
                            n_fft=256, hop_length=32,
                            target_shape=self.stft_shape)
        x_freq = torch.from_numpy(stft).float()

        # Physics features (~30,)
        phys = compute_physics_features(signal, rpm, fs, ff)
        x_phys = torch.from_numpy(phys).float()

        return x_time, x_freq, x_phys, torch.tensor(label, dtype=torch.long)
