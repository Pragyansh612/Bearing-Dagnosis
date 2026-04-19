import numpy as np

def augment_signal(signal, rpm, fs, noise_snr_db=25.0, 
                   amp_scale_range=(0.8, 1.2), 
                   rpm_jitter=0.05,
                   apply_prob=0.5):
    """Apply random augmentations to signal"""
    signal = signal.copy()
    rpm = float(rpm)

    # Random amplitude scaling
    if np.random.rand() < apply_prob:
        scale = np.random.uniform(*amp_scale_range)
        signal *= scale

    # Add Gaussian noise
    if np.random.rand() < apply_prob:
        rms = np.sqrt(np.mean(signal ** 2))
        snr_linear = 10 ** (noise_snr_db / 10.0)
        noise_rms = rms / np.sqrt(snr_linear)
        noise = np.random.normal(0, noise_rms, len(signal))
        signal += noise.astype(np.float32)

    # RPM jitter for physics robustness
    if np.random.rand() < apply_prob:
        jitter = np.random.uniform(-rpm_jitter, rpm_jitter)
        rpm = rpm * (1 + jitter)

    # Random time shift
    if np.random.rand() < apply_prob:
        shift = np.random.randint(-50, 51)
        signal = np.roll(signal, shift)

    return signal, rpm
