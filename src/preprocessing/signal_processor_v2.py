import numpy as np
from scipy.signal import spectrogram
from scipy.ndimage import zoom

def compute_physics_features(signal, rpm, fs, fault_freq):
    """Original physics features — unchanged"""
    from src.preprocessing.signal_processor import compute_physics_features as _orig
    return _orig(signal, rpm, fs, fault_freq)

def normalize_signal(signal):
    mean = signal.mean()
    std  = signal.std() + 1e-8
    return ((signal - mean) / std).astype(np.float32)

def compute_order_spectrum(signal, rpm, fs, n_orders=100, order_resolution=0.1):
    """
    Convert frequency spectrum to ORDER domain.
    order = frequency / shaft_frequency
    This makes fault signatures appear at SAME order
    regardless of RPM or bearing geometry.
    
    BPFO always at BPFOMultiple order
    BPFI always at BPFIMultiple order
    BSF  always at BPFMultiple order
    FTF  always at FTFMultiple order
    """
    shaft_freq = rpm / 60.0
    if shaft_freq <= 0:
        shaft_freq = 20.0  # fallback 1200 RPM

    # FFT of signal
    N       = len(signal)
    fft_mag = np.abs(np.fft.rfft(signal)) / N
    freqs   = np.fft.rfftfreq(N, d=1.0/fs)

    # Convert to order axis
    max_order   = min(n_orders, (fs/2) / shaft_freq)
    order_axis  = np.arange(0, max_order, order_resolution)
    order_spectrum = np.zeros(len(order_axis), dtype=np.float32)

    for i, order in enumerate(order_axis):
        freq_hz = order * shaft_freq
        if freq_hz >= fs/2:
            break
        # Find nearest FFT bin
        idx = np.argmin(np.abs(freqs - freq_hz))
        # Interpolate between bins
        if idx > 0 and idx < len(fft_mag)-1:
            order_spectrum[i] = float(fft_mag[idx])
        else:
            order_spectrum[i] = float(fft_mag[min(idx, len(fft_mag)-1)])

    # Normalize
    mean = order_spectrum.mean()
    std  = order_spectrum.std() + 1e-8
    order_spectrum = (order_spectrum - mean) / std

    return order_spectrum.astype(np.float32)

def compute_order_stft(signal, rpm, fs, target_size=(64, 64)):
    """
    Order-domain STFT (Order Tracking Spectrogram).
    Y-axis = orders (multiples of shaft frequency)
    X-axis = time
    This is machine-invariant — same fault appears at same Y position
    regardless of which machine.
    """
    shaft_freq = rpm / 60.0
    if shaft_freq <= 0:
        shaft_freq = 20.0

    # Compute regular STFT
    f, t, Sxx = spectrogram(signal, fs=fs, nperseg=256, noverlap=224)
    Sxx_db = 10 * np.log10(np.abs(Sxx) + 1e-10)

    # Convert frequency axis to order axis
    order_axis = f / shaft_freq  # (n_freqs,)

    # Resample to uniform order grid (0 to max_order)
    max_order = min(50.0, order_axis[-1])  # up to 50x shaft frequency
    target_orders = np.linspace(0, max_order, target_size[0])

    # Interpolate each time slice from freq to order domain
    order_stft = np.zeros((target_size[0], Sxx_db.shape[1]), dtype=np.float32)
    for j in range(Sxx_db.shape[1]):
        order_stft[:, j] = np.interp(target_orders, order_axis, Sxx_db[:, j])

    # Resize time axis to target_size[1]
    if order_stft.shape[1] != target_size[1]:
        zt = target_size[1] / order_stft.shape[1]
        order_stft = zoom(order_stft, (1.0, zt))

    # Normalize to [0,1]
    mn = order_stft.min()
    mx = order_stft.max()
    if mx - mn > 1e-8:
        order_stft = (order_stft - mn) / (mx - mn)

    return order_stft.astype(np.float32)

def compute_order_physics_features(signal, rpm, fs, fault_freq):
    """
    Physics features in ORDER domain.
    Instead of amplitude at Hz, extract amplitude at ORDER positions.
    Makes features invariant to RPM and bearing geometry differences.
    """
    shaft_freq = rpm / 60.0
    if shaft_freq <= 0:
        shaft_freq = 20.0

    N       = len(signal)
    fft_mag = np.abs(np.fft.rfft(signal)) / N
    freqs   = np.fft.rfftfreq(N, d=1.0/fs)

    def amp_at_order(order):
        """Get FFT amplitude at given order (multiple of shaft frequency)"""
        freq_hz = order * shaft_freq
        if freq_hz <= 0 or freq_hz >= fs/2:
            return 0.0
        idx = np.argmin(np.abs(freqs - freq_hz))
        return float(fft_mag[idx])

    # Fault characteristic orders (from multipliers)
    ftf_order  = fault_freq['FTFMultiple']
    bpf_order  = fault_freq['BPFMultiple']
    bpfo_order = fault_freq['BPFOMultiple']
    bpfi_order = fault_freq['BPFIMultiple']

    features = []

    # FTF harmonics (4)
    for h in [1, 2, 3, 4]:
        features.append(amp_at_order(ftf_order * h))

    # BPF (ball) harmonics (4)
    for h in [1, 2, 3, 4]:
        features.append(amp_at_order(bpf_order * h))

    # BPFO (outer race) harmonics (4)
    for h in [1, 2, 3, 4]:
        features.append(amp_at_order(bpfo_order * h))

    # BPFI (inner race) harmonics (4)
    for h in [1, 2, 3, 4]:
        features.append(amp_at_order(bpfi_order * h))

    # BPFO sidebands (2) — at BPFO ± 1 order
    features.append(amp_at_order(bpfo_order - 1.0))
    features.append(amp_at_order(bpfo_order + 1.0))

    # BPFI sidebands (2)
    features.append(amp_at_order(bpfi_order - 1.0))
    features.append(amp_at_order(bpfi_order + 1.0))

    # Statistical features (same as original — 6)
    std = signal.std() + 1e-8
    rms    = float(np.sqrt(np.mean(signal**2)))
    peak   = float(np.abs(signal).max())
    crest  = peak / (rms + 1e-8)
    kurt   = float(np.mean(((signal - signal.mean())/std)**4))
    skew   = float(np.mean(((signal - signal.mean())/std)**3))
    p2p    = float(signal.max() - signal.min())
    features.extend([rms, peak, crest, kurt, skew, p2p])

    # Order band energies (4) — energy in bands of orders
    order_bands = [(0, 2), (2, 5), (5, 10), (10, 20)]
    order_spec  = compute_order_spectrum(signal, rpm, fs, n_orders=20)
    order_axis  = np.arange(0, 20, 0.1)[:len(order_spec)]
    for lo, hi in order_bands:
        mask = (order_axis >= lo) & (order_axis < hi)
        band_energy = float(order_spec[mask].sum()) if mask.sum() > 0 else 0.0
        features.append(band_energy)

    return np.array(features, dtype=np.float32)  # still 30-dim
