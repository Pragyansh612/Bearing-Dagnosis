import streamlit as st
import numpy as np
import torch
import joblib
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import sys, os

sys.path.insert(0, os.path.dirname(__file__))

from src.models.time_branch import TimeDomainBranch
from src.models.freq_branch import FrequencyBranch
from src.models.physics_branch import PhysicsBranch
from src.models.fusion import AttentionFusion
from src.models.autoencoder import BearingAutoencoder
from src.preprocessing.signal_processor_v2 import (
    compute_order_stft, compute_order_physics_features, normalize_signal
)

# ── Config ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="BearingAI — Fault Diagnosis",
    page_icon="⚙️",
    layout="wide"
)

MODELS_DIR  = os.path.join(os.path.dirname(__file__), 'models')
CLASS_NAMES = ['Healthy', 'Inner Race Fault', 'Ball Fault', 'Outer Race Fault']
FAULT_ENERGY_INDICES = {
    1: list(range(12,16)) + [18,19],
    2: list(range(4,8)),
    3: list(range(8,12)) + [16,17],
}
WINDOW_SIZE = 2048
DEVICE      = torch.device('cpu')  # CPU for demo

# ── Load models (cached) ───────────────────────────────────────────────────

@st.cache_resource
def load_models():
    class FinalModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.time_branch    = TimeDomainBranch(out_dim=256)
            self.freq_branch    = FrequencyBranch(out_dim=256)
            self.physics_branch = PhysicsBranch(input_dim=30, out_dim=256)
            self.fusion         = AttentionFusion(256, 3, 4)
        def forward(self, x_time, x_freq, x_phys):
            ft = self.time_branch(x_time)
            ff = self.freq_branch(x_freq)
            fp = self.physics_branch(x_phys)
            return self.fusion(ft, ff, fp)
        def get_fused(self, x_time, x_freq, x_phys):
            ft = self.time_branch(x_time)
            ff = self.freq_branch(x_freq)
            fp = self.physics_branch(x_phys)
            concat  = torch.cat([ft,ff,fp], dim=-1)
            weights = self.fusion.attention(concat)
            stacked = torch.stack([ft,ff,fp], dim=1)
            return (stacked * weights.unsqueeze(-1)).sum(dim=1)

    model = FinalModel()
    ckpt  = torch.load(os.path.join(MODELS_DIR, 'best_model_final_best.pth'),
                       map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    scaler = joblib.load(os.path.join(MODELS_DIR, 'physics_scaler_final_best.pkl'))

    ae = BearingAutoencoder(input_dim=256, latent_dim=32)
    ae_ckpt = torch.load(os.path.join(MODELS_DIR, 'ae_final_best.pth'),
                          map_location='cpu')
    ae.load_state_dict(ae_ckpt['model_state_dict'])
    ae.mean_error = ae_ckpt['mean_error']
    ae.std_error  = ae_ckpt['std_error']
    ae.threshold  = ae_ckpt['threshold']
    ae.eval()

    class_centers = np.load(os.path.join(MODELS_DIR, 'class_centers.npy'))
    baselines     = joblib.load(os.path.join(MODELS_DIR, 'folder_baselines.pkl'))

    return model, scaler, ae, class_centers, baselines

# ── Inference ──────────────────────────────────────────────────────────────

def run_inference(signal, rpm, fs, fault_freq, model, scaler, ae, class_centers, baseline=None):
    rms = float(np.sqrt(np.mean(signal**2)))

    # Physics features
    phys_raw = compute_order_physics_features(signal, rpm, fs, fault_freq)

    # Delta normalize if baseline available
    if baseline is not None:
        sig_proc   = ((signal - baseline['signal_mean']) / baseline['signal_std']).astype(np.float32)
        phys_delta = ((phys_raw - np.array(baseline['physics_mean'])) /
                      np.array(baseline['physics_std'])).astype(np.float32)
    else:
        sig_proc   = normalize_signal(signal)
        phys_delta = phys_raw.copy()

    phys_sc = scaler.transform(phys_delta.reshape(1,-1)).astype(np.float32)

    x_time = torch.from_numpy(sig_proc).unsqueeze(0)
    x_freq = torch.from_numpy(compute_order_stft(sig_proc, rpm, fs)).unsqueeze(0)
    x_phys = torch.from_numpy(phys_sc)

    # MC Dropout uncertainty
    def enable_dropout(m):
        if isinstance(m, torch.nn.Dropout): m.train()

    model.eval()
    model.apply(enable_dropout)
    mc_preds, mc_weights = [], []
    xb_t = x_time.repeat(16,1)
    xb_f = x_freq.repeat(16,1,1)
    xb_p = x_phys.repeat(16,1)
    with torch.no_grad():
        for _ in range(30):
            logits, weights = model(xb_t, xb_f, xb_p)
            mc_preds.append(torch.softmax(logits,dim=-1).numpy()[0])
            mc_weights.append(weights.numpy()[0])

    mc_preds     = np.array(mc_preds)
    mean_pred    = mc_preds.mean(axis=0)
    pred_class   = int(mean_pred.argmax())
    confidence   = float(mean_pred[pred_class])
    uncertainty  = float(mc_preds.std(axis=0)[pred_class])
    mean_weights = np.array(mc_weights).mean(axis=0)

    # AE anomaly score
    model.eval()
    with torch.no_grad():
        fused = model.get_fused(x_time, x_freq, x_phys)
    ae.eval()
    with torch.no_grad():
        ae_error = float(ae.normalized_error(fused).item())

    # Physics energy check
    fault_energy = 0.0
    physics_ok   = True
    if pred_class != 0:
        indices      = FAULT_ENERGY_INDICES.get(pred_class, [])
        fault_energy = float(phys_raw[indices].mean()) if indices else 0.0
        relative     = fault_energy / (rms + 1e-8)
        physics_ok   = relative > 0.01

    # Prototype distance
    fused_np = fused.numpy()[0]
    distances = np.linalg.norm(class_centers - fused_np, axis=1)
    proto_class = int(distances.argmin())

    # BPFO energy for health index
    bpfo_energy = float(phys_raw[8:12].mean())
    bpfi_energy = float(phys_raw[12:16].mean())
    bpf_energy  = float(phys_raw[4:8].mean())
    health_index = max(0.0, 1.0 - min(1.0, ae_error / 20.0))

    # Decision
    if ae_error < 5.0:
        decision = 'HEALTHY'
        status_color = 'green'
    elif uncertainty > 0.15:
        decision = 'UNCERTAIN'
        status_color = 'orange'
    elif not physics_ok and pred_class != 0:
        decision = 'PHYSICS MISMATCH'
        status_color = 'orange'
    elif ae_error > 10.0 and uncertainty > 0.15:
        decision = 'UNKNOWN FAULT'
        status_color = 'red'
    else:
        decision = 'FAULT DETECTED'
        status_color = 'red'

    return {
        'pred_class':    pred_class,
        'pred_name':     CLASS_NAMES[pred_class],
        'proto_class':   proto_class,
        'proto_name':    CLASS_NAMES[proto_class],
        'confidence':    confidence,
        'uncertainty':   uncertainty,
        'ae_error':      ae_error,
        'health_index':  health_index,
        'physics_ok':    physics_ok,
        'fault_energy':  fault_energy,
        'bpfo_energy':   bpfo_energy,
        'bpfi_energy':   bpfi_energy,
        'bpf_energy':    bpf_energy,
        'decision':      decision,
        'status_color':  status_color,
        'branch_weights': mean_weights,
        'mean_pred':     mean_pred,
        'all_probs':     {CLASS_NAMES[i]: float(mean_pred[i]) for i in range(4)},
        'signal':        signal,
        'rpm':           rpm,
        'fs':            fs,
        'rms':           rms,
        'phys_raw':      phys_raw,
    }

# ── Signal loading helpers ─────────────────────────────────────────────────

def safe_scalar(val):
    return float(np.array(val).flat[0])

def extract_signal(row):
    arr = row
    while True:
        if isinstance(arr, np.ndarray):
            if arr.dtype.kind in ('f','i','u'):
                return arr.flatten().astype(np.float32)
            elif arr.dtype.kind == 'O':
                if arr.size == 1: arr = arr.flat[0]
                else:
                    parts = []
                    for item in arr.flat:
                        try: parts.append(extract_signal(item))
                        except: pass
                    if parts: return np.concatenate(parts).astype(np.float32)
                    raise ValueError()
            else: return arr.flatten().astype(np.float32)
        elif isinstance(arr,(int,float)): return np.array([arr],dtype=np.float32)
        else: raise ValueError()

def get_source(mat, fault_origin):
    for key in [fault_origin,'DS','FS','Upper','Lower']:
        if key in mat and isinstance(mat[key],dict) and 'rawData' in mat[key]:
            return mat[key]
    return None

def load_mat_signal(mat_bytes, meas_idx=0):
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as tmp:
        tmp.write(mat_bytes)
        tmp_path = tmp.name
    mat = sio.loadmat(tmp_path, simplify_cells=True)
    os.unlink(tmp_path)

    fault_type   = int(mat.get('faultType', 0))
    fault_origin = str(mat.get('faultOrigin', 'DS'))
    src = get_source(mat, fault_origin)
    if src is None: return None

    raw_data = np.array(src['rawData'])
    rpm_vals = np.array(src['RPM']).flatten()
    fs_vals  = np.array(src['samplingRate']).flatten()
    ff       = src['faultFrequencies']
    fault_freq = {
        'FTFMultiple':  safe_scalar(ff['FTFMultiple']),
        'BPFMultiple':  safe_scalar(ff['BPFMultiple']),
        'BPFOMultiple': safe_scalar(ff['BPFOMultiple']),
        'BPFIMultiple': safe_scalar(ff['BPFIMultiple']),
    }

    if raw_data.ndim == 2 and raw_data.dtype.kind in ('f','i','u'):
        i   = min(meas_idx, raw_data.shape[0]-1)
        sig = raw_data[i].astype(np.float32)
        rpm = float(rpm_vals[i]) if i < len(rpm_vals) else 1200.0
        fs  = float(fs_vals[i])  if i < len(fs_vals)  else 5120.0
    else:
        flat = raw_data.flatten()
        i    = min(meas_idx, len(flat)-1)
        sig  = extract_signal(flat[i])
        rpm  = float(rpm_vals[i]) if i < len(rpm_vals) else 1200.0
        fs   = float(fs_vals[i])  if i < len(fs_vals)  else 5120.0

    mid = len(sig)//2 - WINDOW_SIZE//2
    window = sig[mid:mid+WINDOW_SIZE]
    return window, fault_type, rpm, fs, fault_freq

# ── Plots ──────────────────────────────────────────────────────────────────

def plot_signal(signal, fs, title="Vibration Signal"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3))
    fig.patch.set_facecolor('#0e1117')
    for ax in axes:
        ax.set_facecolor('#1a1a2e')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('#444')

    t = np.arange(len(signal)) / fs * 1000
    axes[0].plot(t, signal, color='#00d4ff', linewidth=0.6, alpha=0.8)
    axes[0].set_title('Raw Vibration Signal', color='white', fontsize=11)
    axes[0].set_xlabel('Time (ms)', color='white')
    axes[0].set_ylabel('Amplitude (m/s²)', color='white')

    N       = len(signal)
    fft_mag = np.abs(np.fft.rfft(signal)) / N
    freqs   = np.fft.rfftfreq(N, d=1.0/fs)
    axes[1].plot(freqs[:N//4], fft_mag[:N//4], color='#ff6b35', linewidth=0.7)
    axes[1].set_title('Frequency Spectrum (FFT)', color='white', fontsize=11)
    axes[1].set_xlabel('Frequency (Hz)', color='white')
    axes[1].set_ylabel('Amplitude', color='white')

    plt.tight_layout()
    return fig

def plot_order_spectrum(signal, rpm, fs, fault_freq):
    from src.preprocessing.signal_processor_v2 import compute_order_spectrum
    shaft_freq = rpm / 60.0
    order_spec = compute_order_spectrum(signal, rpm, fs, n_orders=30)
    order_axis = np.arange(0, 30, 0.1)[:len(order_spec)]

    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#1a1a2e')
    ax.tick_params(colors='white')
    for spine in ax.spines.values(): spine.set_color('#444')

    ax.plot(order_axis, order_spec, color='#a8ff78', linewidth=0.8)

    # Mark fault frequency orders
    colors = {'BPFO': '#ff4444', 'BPFI': '#ffaa00', 'BSF': '#00aaff', 'FTF': '#aa44ff'}
    labels = {
        'BPFO': fault_freq['BPFOMultiple'],
        'BPFI': fault_freq['BPFIMultiple'],
        'BSF':  fault_freq['BPFMultiple'],
        'FTF':  fault_freq['FTFMultiple'],
    }
    for name, order in labels.items():
        if order < 30:
            ax.axvline(x=order, color=colors[name], linestyle='--',
                       alpha=0.8, linewidth=1.2, label=f'{name} ({order:.2f}×)')

    ax.set_title('Order Domain Spectrum — Machine Invariant', color='white', fontsize=11)
    ax.set_xlabel('Order (× shaft frequency)', color='white')
    ax.set_ylabel('Normalized amplitude', color='white')
    ax.legend(facecolor='#1a1a2e', edgecolor='#444', labelcolor='white', fontsize=8)
    plt.tight_layout()
    return fig

def plot_branch_weights(weights):
    fig, ax = plt.subplots(figsize=(5, 2.5))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#1a1a2e')
    ax.tick_params(colors='white')
    for spine in ax.spines.values(): spine.set_color('#444')

    branches = ['Time\nDomain', 'Frequency\nDomain', 'Physics\nMLP']
    colors   = ['#00d4ff', '#ff6b35', '#a8ff78']
    bars     = ax.bar(branches, weights, color=colors, alpha=0.8, width=0.5)
    for bar, w in zip(bars, weights):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{w:.2f}', ha='center', va='bottom', color='white', fontsize=9)
    ax.set_title('Branch Attention Weights', color='white', fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Weight', color='white')
    plt.tight_layout()
    return fig

def plot_class_probs(probs):
    fig, ax = plt.subplots(figsize=(5, 2.5))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#1a1a2e')
    ax.tick_params(colors='white')
    for spine in ax.spines.values(): spine.set_color('#444')

    classes = list(probs.keys())
    values  = list(probs.values())
    short   = ['Healthy', 'Inner\nRace', 'Ball', 'Outer\nRace']
    colors  = ['#2ecc71' if v == max(values) else '#555' for v in values]
    bars    = ax.bar(short, values, color=colors, alpha=0.85, width=0.5)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{v:.2f}', ha='center', va='bottom', color='white', fontsize=8)
    ax.set_title('Class Probabilities (MC Dropout)', color='white', fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Probability', color='white')
    plt.tight_layout()
    return fig

# ── Main UI ────────────────────────────────────────────────────────────────

def main():
    # Header
    st.markdown("""
    <div style='background: linear-gradient(90deg, #0f0c29, #302b63, #24243e);
                padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
        <h1 style='color: #00d4ff; margin:0; font-size: 2rem;'>⚙️ BearingAI</h1>
        <p style='color: #aaa; margin:0; font-size: 1rem;'>
            Multimodal Physics-Informed Bearing Fault Diagnosis System
        </p>
        <p style='color: #666; margin:0; font-size: 0.8rem;'>
            Order-Domain Features · Cross-Machine Generalization · Physics Consistency · Uncertainty Quantification
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Load models
    with st.spinner("Loading models..."):
        model, scaler, ae, class_centers, baselines = load_models()
    st.success("Models loaded", icon="✅")

    # Sidebar
    st.sidebar.title("⚙️ Input")
    input_mode = st.sidebar.radio("Input Mode", ["Upload .mat file", "Demo samples"])

    signal     = None
    fault_type = None
    rpm        = 1200.0
    fs         = 5120.0
    fault_freq = None
    baseline   = None
    source_name = ""

    if input_mode == "Upload .mat file":
        uploaded = st.sidebar.file_uploader("Upload .mat file", type=['mat'])
        meas_idx = st.sidebar.number_input("Measurement index", 0, 500, 0)
        if uploaded:
            try:
                result = load_mat_signal(uploaded.read(), meas_idx)
                if result:
                    signal, fault_type, rpm, fs, fault_freq = result
                    source_name = uploaded.name
                    st.sidebar.success(f"Loaded: {len(signal)} samples")
                    st.sidebar.info(f"RPM: {rpm:.0f} | FS: {fs:.0f} Hz")
            except Exception as e:
                st.sidebar.error(f"Error: {e}")

    else:
        folder = st.sidebar.selectbox(
            "Select folder (hidden test machine)",
            ["Folder 8 — Outer Race Fault",
             "Folder 9 — Outer Race Fault",
             "Folder 10 — Outer Race Fault"]
        )
        mat_file = st.sidebar.selectbox("File", ["train.mat (healthy)", "test.mat (fault)"])
        meas_idx = st.sidebar.slider("Measurement", 0, 50, 5)

        folder_num = folder.split(" ")[1]
        mat_name   = mat_file.split(" ")[0]

        # Look for dataset
        dataset_paths = [
            os.path.expanduser(f'~/SCA bearing dataset/data/{folder_num}/{mat_name}'),
            os.path.expanduser(f'~/bearing_dataset/data/{folder_num}/{mat_name}'),
            f'/data/SCA bearing dataset/data/{folder_num}/{mat_name}',
        ]
        mat_path = None
        for p in dataset_paths:
            if os.path.exists(p):
                mat_path = p
                break

        if mat_path:
            try:
                with open(mat_path, 'rb') as f:
                    result = load_mat_signal(f.read(), meas_idx)
                if result:
                    signal, fault_type, rpm, fs, fault_freq = result
                    source_name = f"Folder {folder_num}/{mat_name}"
                    # Use pre-computed baseline if available
                    if folder_num in baselines:
                        baseline = baselines[folder_num]
                    st.sidebar.success(f"Loaded from folder {folder_num}")
                    st.sidebar.info(f"RPM: {rpm:.0f} | FS: {fs:.0f} Hz")
                    if fault_type is not None:
                        st.sidebar.info(f"True label: {CLASS_NAMES[fault_type]}")
            except Exception as e:
                st.sidebar.error(f"Dataset not found at expected path. Please upload .mat file directly.")
        else:
            st.sidebar.warning("Dataset not found on this machine. Please upload a .mat file.")

    # Manual RPM override
    rpm_override = st.sidebar.number_input("Override RPM (0 = auto)", 0, 5000, 0)
    if rpm_override > 0 and signal is not None:
        rpm = float(rpm_override)

    run_btn = st.sidebar.button("🔍 Run Diagnosis", type="primary",
                                 disabled=(signal is None))

    # Main content
    if signal is None:
        st.info("👈 Upload a .mat file or select a demo sample to begin.")
        st.markdown("""
        ### How it works
        1. **Upload** a bearing vibration signal (.mat format) or **select a demo sample**
        2. The system runs a **3-branch multimodal neural network** with order-domain features
        3. **Physics consistency** is verified against bearing fault frequencies
        4. **MC Dropout** quantifies prediction uncertainty (50 passes)
        5. **Autoencoder** detects unknown/unseen fault patterns
        
        ### Model Performance
        | Dataset | Accuracy | Macro F1 |
        |---------|----------|----------|
        | Internal (folders 1-7) | 98-99% | 0.98-0.99 |
        | Unseen machines (folders 8-10) | 63.7% | 0.425 |
        """)
        return

    # Show signal preview
    st.subheader(f"📡 Signal: {source_name}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Samples", f"{len(signal):,}")
    col2.metric("RPM", f"{rpm:.0f}")
    col3.metric("Sampling Rate", f"{fs:.0f} Hz")
    col4.metric("Duration", f"{len(signal)/fs*1000:.1f} ms")

    fig_sig = plot_signal(signal, fs)
    st.pyplot(fig_sig)
    plt.close()

    if not run_btn:
        return

    # Run inference
    with st.spinner("Running diagnosis..."):
        result = run_inference(signal, rpm, fs, fault_freq, model, scaler,
                               ae, class_centers, baseline)

    st.markdown("---")
    st.subheader("🧠 Diagnosis Result")

    # Status color mapping
    color_map = {'green': '#2ecc71', 'orange': '#f39c12', 'red': '#e74c3c'}
    status_color = color_map.get(result['status_color'], '#aaa')

    # Main result card
    st.markdown(f"""
    <div style='background: #1a1a2e; border-left: 5px solid {status_color};
                padding: 20px; border-radius: 8px; margin-bottom: 20px;'>
        <h2 style='color: {status_color}; margin:0;'>
            {result['pred_name']}
        </h2>
        <p style='color: #aaa; margin:5px 0 0 0; font-size: 1.1rem;'>
            {result['decision']}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Metrics row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Confidence", f"{result['confidence']:.1%}")
    c2.metric("Uncertainty", f"±{result['uncertainty']:.3f}")
    c3.metric("AE Anomaly Score", f"{result['ae_error']:.1f}σ")
    c4.metric("Health Index", f"{result['health_index']:.2f}")
    c5.metric("Prototype Match", result['proto_name'])

    # Health bar
    health = result['health_index']
    health_color = '#2ecc71' if health > 0.7 else '#f39c12' if health > 0.4 else '#e74c3c'
    health_label = 'Good' if health > 0.7 else 'Degrading' if health > 0.4 else 'Critical'
    st.markdown(f"""
    <div style='margin: 10px 0;'>
        <div style='display:flex; justify-content:space-between; color:#aaa; font-size:0.85rem;'>
            <span>Machine Health</span><span>{health_label} ({health:.0%})</span>
        </div>
        <div style='background:#333; border-radius:10px; height:12px; margin-top:4px;'>
            <div style='background:{health_color}; width:{health*100:.0f}%;
                        height:12px; border-radius:10px;'></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Two columns: explanation + charts
    left, right = st.columns([1, 1])

    with left:
        st.subheader("🔬 Physics Analysis")

        # Physics explanation
        explanation_items = []
        bpfo = result['bpfo_energy']
        bpfi = result['bpfi_energy']
        bpf  = result['bpf_energy']

        if bpfo > 0.1:
            explanation_items.append(f"⚠️ Elevated BPFO energy ({bpfo:.4f}) → outer race stress")
        else:
            explanation_items.append(f"✅ Normal BPFO energy ({bpfo:.4f})")

        if bpfi > 0.1:
            explanation_items.append(f"⚠️ Elevated BPFI energy ({bpfi:.4f}) → inner race stress")
        else:
            explanation_items.append(f"✅ Normal BPFI energy ({bpfi:.4f})")

        if bpf > 0.1:
            explanation_items.append(f"⚠️ Elevated BSF energy ({bpf:.4f}) → ball fault stress")
        else:
            explanation_items.append(f"✅ Normal BSF energy ({bpf:.4f})")

        if result['physics_ok']:
            explanation_items.append("✅ Physics consistency: PASSED")
        else:
            explanation_items.append("⚠️ Physics consistency: MISMATCH")

        if result['uncertainty'] > 0.15:
            explanation_items.append(f"⚠️ High uncertainty (±{result['uncertainty']:.3f}) — domain shift possible")
        else:
            explanation_items.append(f"✅ Low uncertainty (±{result['uncertainty']:.3f})")

        if result['ae_error'] > 5.0:
            explanation_items.append(f"⚠️ AE score {result['ae_error']:.1f}σ — anomalous pattern")
        else:
            explanation_items.append(f"✅ AE score {result['ae_error']:.1f}σ — normal pattern")

        for item in explanation_items:
            color = '#2ecc71' if item.startswith('✅') else '#f39c12'
            st.markdown(f"<p style='color:{color}; margin:4px 0;'>{item}</p>",
                        unsafe_allow_html=True)

        # Maintenance recommendation
        st.subheader("🔧 Recommendation")
        recs = {
            'HEALTHY':          ("Continue normal operation. Schedule routine inspection.", "green"),
            'FAULT DETECTED':   ("Schedule bearing replacement within 24–72 hours.", "red"),
            'UNCERTAIN':        ("Recommend physical inspection within 24 hours.", "orange"),
            'PHYSICS MISMATCH': ("Anomaly detected. Physical inspection recommended.", "orange"),
            'UNKNOWN FAULT':    ("Unknown fault pattern. Stop machine and inspect immediately.", "red"),
        }
        rec_text, rec_color = recs.get(result['decision'],
                                        ("Schedule inspection.", "orange"))
        st.markdown(f"""
        <div style='background:#1a1a2e; border-left:4px solid {color_map[rec_color]};
                    padding:12px; border-radius:6px;'>
            <p style='color:{color_map[rec_color]}; margin:0;'>{rec_text}</p>
        </div>
        """, unsafe_allow_html=True)

        if fault_type is not None:
            true_name = CLASS_NAMES[fault_type]
            correct   = (result['pred_class'] == fault_type)
            icon      = "✅" if correct else "❌"
            st.markdown(f"""
            <div style='margin-top:15px; padding:10px; background:#1a1a2e; border-radius:6px;'>
                <p style='color:#aaa; margin:0; font-size:0.85rem;'>
                    {icon} True label: <strong style='color:white;'>{true_name}</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)

    with right:
        st.subheader("📊 Model Internals")

        fig_weights = plot_branch_weights(result['branch_weights'])
        st.pyplot(fig_weights)
        plt.close()

        fig_probs = plot_class_probs(result['all_probs'])
        st.pyplot(fig_probs)
        plt.close()

    # Order spectrum
    st.markdown("---")
    st.subheader("📈 Order Domain Spectrum")
    st.caption("Fault frequencies marked — invariant to RPM and machine geometry")
    fig_order = plot_order_spectrum(signal, rpm, fs, fault_freq)
    st.pyplot(fig_order)
    plt.close()

    # Technical details expander
    with st.expander("🔧 Technical Details"):
        col1, col2 = st.columns(2)
        col1.write(f"**Shaft frequency:** {rpm/60:.2f} Hz")
        col1.write(f"**BPFO order:** {fault_freq['BPFOMultiple']:.3f}×")
        col1.write(f"**BPFI order:** {fault_freq['BPFIMultiple']:.3f}×")
        col1.write(f"**BSF order:** {fault_freq['BPFMultiple']:.3f}×")
        col1.write(f"**FTF order:** {fault_freq['FTFMultiple']:.3f}×")
        col2.write(f"**Signal RMS:** {result['rms']:.6f}")
        col2.write(f"**AE mean error (train):** {ae.mean_error:.6f}")
        col2.write(f"**AE threshold (5σ):** {ae.threshold:.6f}")
        col2.write(f"**Raw prediction:** {result['pred_name']} ({result['confidence']:.3f})")
        col2.write(f"**Prototype prediction:** {result['proto_name']}")

if __name__ == '__main__':
    main()