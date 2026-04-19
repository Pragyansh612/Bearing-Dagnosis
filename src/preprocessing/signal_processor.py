import numpy as np
import scipy.signal as scipy_signal
from scipy.fft import rfft, rfftfreq

def normalize_signal(signal):
    std = signal.std()
    if std < 1e-8:
        return signal
    return (signal - signal.mean()) / std

def compute_fft(signal, fs):
    """Returns magnitude spectrum and frequency array (in orders if rpm given)"""
    n = len(signal)
    fft_mag = np.abs(rfft(signal)) / n
    freqs = rfftfreq(n, 1.0 / fs)
    return fft_mag.astype(np.float32), freqs.astype(np.float32)

def compute_stft(signal, fs, n_fft=256, hop_length=32, target_shape=(64, 64)):
    """Returns 2D spectrogram (target_shape) in dB"""
    from scipy.signal import spectrogram as sp_spectrogram
    freqs, times, Sxx = sp_spectrogram(
        signal, fs=fs, nperseg=n_fft, noverlap=n_fft - hop_length,
        scaling='spectrum'
    )
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    # Resize to target_shape
    from scipy.ndimage import zoom
    zoom_r = target_shape[0] / Sxx_db.shape[0]
    zoom_c = target_shape[1] / Sxx_db.shape[1]
    Sxx_resized = zoom(Sxx_db, (zoom_r, zoom_c), order=1)
    # Normalize to [0, 1]
    mn, mx = Sxx_resized.min(), Sxx_resized.max()
    if mx > mn:
        Sxx_resized = (Sxx_resized - mn) / (mx - mn)
    return Sxx_resized.astype(np.float32)

def get_amplitude_at_freq(fft_mag, freqs, target_freq, tolerance_hz=2.0):
    """Extract amplitude at a specific frequency"""
    idx = np.argmin(np.abs(freqs - target_freq))
    # Average over small band
    band = np.abs(freqs - target_freq) <= tolerance_hz
    if band.sum() > 0:
        return float(fft_mag[band].max())
    return float(fft_mag[idx])

def compute_physics_features(signal, rpm, fs, fault_freqs):
    """
    Compute physics-informed features:
    - Amplitudes at fault characteristic frequencies (fundamental + 3 harmonics)
    - Sideband amplitudes
    - Statistical features
    Returns: numpy array of shape (n_features,)
    """
    shaft_freq = rpm / 60.0
    if shaft_freq <= 0:
        shaft_freq = 1.0

    # Convert multiples to actual Hz
    ftf_hz  = fault_freqs['FTFMultiple']  * shaft_freq
    bpf_hz  = fault_freqs['BPFMultiple']  * shaft_freq
    bpfo_hz = fault_freqs['BPFOMultiple'] * shaft_freq
    bpfi_hz = fault_freqs['BPFIMultiple'] * shaft_freq

    fft_mag, freqs = compute_fft(signal, fs)
    tol = fs / len(signal) * 3  # 3 bins tolerance

    features = []

    # Amplitudes at fundamentals + 3 harmonics for each fault freq
    for base_freq in [ftf_hz, bpf_hz, bpfo_hz, bpfi_hz]:
        if base_freq <= 0:
            features.extend([0.0] * 4)
            continue
        for harmonic in [1, 2, 3, 4]:
            freq = base_freq * harmonic
            if freq < fs / 2:
                features.append(get_amplitude_at_freq(fft_mag, freqs, freq, tol))
            else:
                features.append(0.0)

    # Sidebands: ±shaft_freq around BPFO and BPFI
    for base_freq in [bpfo_hz, bpfi_hz]:
        if base_freq <= 0:
            features.extend([0.0, 0.0])
            continue
        for delta in [-shaft_freq, shaft_freq]:
            sb_freq = base_freq + delta
            if 0 < sb_freq < fs / 2:
                features.append(get_amplitude_at_freq(fft_mag, freqs, sb_freq, tol))
            else:
                features.append(0.0)

    # Statistical features
    signal_std = signal.std() + 1e-8
    rms = float(np.sqrt(np.mean(signal ** 2)))
    peak = float(np.max(np.abs(signal)))
    crest_factor = peak / (rms + 1e-8)
    kurtosis = float(np.mean(((signal - signal.mean()) / signal_std) ** 4))
    skewness = float(np.mean(((signal - signal.mean()) / signal_std) ** 3))
    peak_to_peak = float(signal.max() - signal.min())

    features.extend([rms, peak, crest_factor, kurtosis, skewness, peak_to_peak])

    # Energy in order bands (orders 0-2, 2-5, 5-10, 10-20)
    order_freqs = freqs / shaft_freq if shaft_freq > 0 else freqs
    for lo, hi in [(0, 2), (2, 5), (5, 10), (10, 20)]:
        band = (order_freqs >= lo) & (order_freqs < hi)
        features.append(float(fft_mag[band].sum()) if band.sum() > 0 else 0.0)

    return np.array(features, dtype=np.float32)

def get_physics_feature_dim():
    """Return dimensionality of physics features"""
    # 4 fault freqs * 4 harmonics = 16
    # 2 fault freqs * 2 sidebands = 4
    # 6 statistical
    # 4 order bands
    return 16 + 4 + 6 + 4  # = 30
