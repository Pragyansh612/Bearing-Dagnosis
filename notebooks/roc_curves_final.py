import os, sys, json
import numpy as np
import torch
import joblib
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize

sys.path.insert(0, os.path.expanduser('~/bearing_diagnosis'))

from src.models.time_branch import TimeDomainBranch
from src.models.freq_branch import FrequencyBranch
from src.models.physics_branch import PhysicsBranch
from src.models.fusion import AttentionFusion
from src.models.autoencoder import BearingAutoencoder
from src.preprocessing.signal_processor_v2 import (
    compute_order_stft, compute_order_physics_features, normalize_signal
)

DATASET_PATH     = os.path.expanduser('~') + '/SCA bearing dataset/data'
MODELS_DIR       = os.path.expanduser('~/bearing_diagnosis/models')
RESULTS_DIR      = os.path.expanduser('~/bearing_diagnosis/results')
DEVICE           = torch.device('cuda')
CLASS_NAMES      = ['Healthy', 'Inner Race Fault', 'Ball Fault', 'Outer Race Fault']
COLORS           = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12']
HIDDEN_FOLDERS   = ['8', '9', '10']
WINDOW_SIZE      = 2048
N_SAMPLES        = 100
N_MC             = 30
BPFO_INDICES             = list(range(8,12)) + [16,17]
FAULT_ENERGY_INDICES     = {1: list(range(12,16))+[18,19], 2: list(range(4,8)), 3: list(range(8,12))+[16,17]}
PHYSICS_ENERGY_THRESHOLD = 0.01
AE_SIGMA_THRESHOLD       = 8.0
HEALTHY_MARGIN_THRESHOLD = 0.08
BPFO_ENERGY_THRESHOLD    = 0.015
KURTOSIS_HEALTHY_MAX     = 3.2
CREST_HEALTHY_MAX        = 4.5

class OrderModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.time_branch    = TimeDomainBranch(out_dim=256)
        self.freq_branch    = FrequencyBranch(out_dim=256)
        self.physics_branch = PhysicsBranch(input_dim=30, out_dim=256)
        self.fusion         = AttentionFusion(256, 3, 4)
    def forward(self, x_time, x_freq, x_phys):
        f_time = self.time_branch(x_time)
        f_freq = self.freq_branch(x_freq)
        f_phys = self.physics_branch(x_phys)
        return self.fusion(f_time, f_freq, f_phys)
    def get_fused(self, x_time, x_freq, x_phys):
        f_time  = self.time_branch(x_time)
        f_freq  = self.freq_branch(x_freq)
        f_phys  = self.physics_branch(x_phys)
        concat  = torch.cat([f_time,f_freq,f_phys],dim=-1)
        weights = self.fusion.attention(concat)
        stacked = torch.stack([f_time,f_freq,f_phys],dim=1)
        return (stacked*weights.unsqueeze(-1)).sum(dim=1)

model   = OrderModel().to(DEVICE)
ckpt    = torch.load(os.path.join(MODELS_DIR, 'best_model_final_best.pth'))
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
scaler  = joblib.load(os.path.join(MODELS_DIR, 'physics_scaler_final_best.pkl'))
centers = np.load(os.path.join(MODELS_DIR, 'class_centers.npy'))
ae = BearingAutoencoder(input_dim=256, latent_dim=32).to(DEVICE)
ae_ckpt = torch.load(os.path.join(MODELS_DIR, 'ae_final_best.pth'))
ae.load_state_dict(ae_ckpt['model_state_dict'])
ae.mean_error = ae_ckpt['mean_error']
ae.std_error  = ae_ckpt['std_error']
ae.threshold  = ae_ckpt['threshold']
ae.eval()

def safe_scalar(val):
    return float(np.array(val).flat[0])

def extract_1d_signal(row):
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
                        try: parts.append(extract_1d_signal(item))
                        except: pass
                    if parts: return np.concatenate(parts).astype(np.float32)
                    raise ValueError()
            else: return arr.flatten().astype(np.float32)
        elif isinstance(arr,(int,float)): return np.array([arr],dtype=np.float32)
        else: raise ValueError()

def get_signal_source(mat, fault_origin):
    for key in [fault_origin,'DS','FS','Upper','Lower']:
        if key in mat and isinstance(mat[key],dict) and 'rawData' in mat[key]:
            return mat[key]
    return None

def load_samples(folder, mat_file, n=100):
    fpath = os.path.join(DATASET_PATH, folder, mat_file)
    if not os.path.exists(fpath): return []
    mat = sio.loadmat(fpath, simplify_cells=True)
    fault_type   = int(mat.get('faultType', 0))
    fault_origin = str(mat.get('faultOrigin', 'DS'))
    src = get_signal_source(mat, fault_origin)
    if src is None: return []
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
    samples = []
    if raw_data.ndim==2 and raw_data.dtype.kind in ('f','i','u'):
        meas_list = [(raw_data[i].astype(np.float32),
                      float(rpm_vals[i]) if i<len(rpm_vals) else 1200.0,
                      float(fs_vals[i])  if i<len(fs_vals)  else 5120.0)
                     for i in range(raw_data.shape[0])]
    else:
        flat = raw_data.flatten()
        meas_list = []
        for i in range(len(flat)):
            try:
                sig = extract_1d_signal(flat[i])
                meas_list.append((sig,
                    float(rpm_vals[i]) if i<len(rpm_vals) else 1200.0,
                    float(fs_vals[i])  if i<len(fs_vals)  else 5120.0))
            except: pass
    for sig, rpm, fs in meas_list:
        if len(sig)<WINDOW_SIZE: continue
        mid = len(sig)//2 - WINDOW_SIZE//2
        w   = sig[mid:mid+WINDOW_SIZE]
        samples.append((w, fault_type, rpm, fs, fault_freq))
        if len(samples) >= n: break
    return samples

def enable_dropout(m):
    if isinstance(m, torch.nn.Dropout): m.train()

def get_probs(signal, rpm, fs, ff):
    rms   = float(np.sqrt(np.mean(signal**2)))
    crest = float(np.abs(signal).max()) / (rms + 1e-8)
    std   = signal.std() + 1e-8
    kurt  = float(np.mean(((signal - signal.mean()) / std) ** 4))
    phys_raw = compute_order_physics_features(signal, rpm, fs, ff)
    phys_sc  = scaler.transform(phys_raw.reshape(1,-1)).astype(np.float32)
    sig_norm = normalize_signal(signal)
    x_time   = torch.from_numpy(sig_norm).unsqueeze(0).to(DEVICE)
    x_freq   = torch.from_numpy(compute_order_stft(sig_norm, rpm, fs)).unsqueeze(0).to(DEVICE)
    x_phys   = torch.from_numpy(phys_sc).to(DEVICE)

    model.eval()
    with torch.no_grad():
        fused = model.get_fused(x_time, x_freq, x_phys)
    ae.eval()
    with torch.no_grad():
        ae_error = float(ae.normalized_error(fused).item())

    emb_np = fused.cpu().numpy()[0]
    emb    = emb_np - emb_np.mean()
    emb    = emb / (emb.std() + 1e-6)
    emb    = emb / (np.linalg.norm(emb) + 1e-8)
    sims   = centers @ emb

    # softmax-like from cosine sims for smooth ROC
    sims_exp  = np.exp(sims * 5.0)
    proto_probs = sims_exp / (sims_exp.sum() + 1e-8)

    model.eval()
    model.apply(enable_dropout)
    mc_preds = []
    xb_t = x_time.repeat(32,1)
    xb_f = x_freq.repeat(32,1,1)
    xb_p = x_phys.repeat(32,1)
    with torch.no_grad():
        for _ in range(N_MC):
            logits, _ = model(xb_t, xb_f, xb_p)
            mc_preds.append(torch.softmax(logits,dim=-1).cpu().numpy()[0])
    mc_probs = np.array(mc_preds).mean(axis=0)

    # Blend: 60% prototype + 40% softmax for smooth probability scores
    blended = 0.6 * proto_probs + 0.4 * mc_probs

    # Apply physics adjustments to scores
    bpfo_energy = float(phys_raw[BPFO_INDICES].mean()) / (rms + 1e-8)
    if bpfo_energy > BPFO_ENERGY_THRESHOLD:
        blended[3] = min(1.0, blended[3] * 1.5)
        blended = blended / blended.sum()

    return blended

print("Collecting probabilities for hidden folders 8-10...")
all_true, all_probs_list = [], []
for folder in HIDDEN_FOLDERS:
    for mat_file in ['train.mat', 'test.mat']:
        samples = load_samples(folder, mat_file, N_SAMPLES)
        if not samples: continue
        true_class = samples[0][1]
        print(f"  Folder {folder}/{mat_file}: {len(samples)} samples, true={CLASS_NAMES[true_class]}")
        for sig, ft, rpm, fs, ff in samples:
            probs = get_probs(sig, rpm, fs, ff)
            all_true.append(ft)
            all_probs_list.append(probs)

l  = np.array(all_true)
pr = np.array(all_probs_list)
present_classes = sorted(np.unique(l).tolist())
print(f"\nClasses present: {[CLASS_NAMES[c] for c in present_classes]}")

# ROC curves
l_bin = label_binarize(l, classes=[0,1,2,3])
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('ROC and Precision-Recall Curves — Hidden Folders 8, 9, 10\n(Unseen Machines — Domain Generalization Test)',
             fontsize=13, fontweight='bold')

ax1 = axes[0]
ax1.set_title('ROC Curves (One-vs-Rest)', fontsize=11)
macro_auc_vals = []
for i in present_classes:
    if l_bin[:, i].sum() == 0: continue
    fpr, tpr, _ = roc_curve(l_bin[:, i], pr[:, i])
    roc_auc_val = auc(fpr, tpr)
    macro_auc_vals.append(roc_auc_val)
    ax1.plot(fpr, tpr, color=COLORS[i], lw=2,
             label=f'{CLASS_NAMES[i]} (AUC={roc_auc_val:.3f})')
ax1.plot([0,1],[0,1],'k--',lw=1,alpha=0.5,label='Random')
ax1.set_xlabel('False Positive Rate', fontsize=10)
ax1.set_ylabel('True Positive Rate', fontsize=10)
ax1.legend(loc='lower right', fontsize=9)
ax1.grid(True, alpha=0.3)
macro_auc = np.mean(macro_auc_vals)
ax1.set_title(f'ROC Curves (One-vs-Rest)\nMacro AUC = {macro_auc:.3f}', fontsize=11)

ax2 = axes[1]
for i in present_classes:
    if l_bin[:, i].sum() == 0: continue
    prec_vals, rec_vals, _ = precision_recall_curve(l_bin[:, i], pr[:, i])
    pr_auc = auc(rec_vals, prec_vals)
    ax2.plot(rec_vals, prec_vals, color=COLORS[i], lw=2,
             label=f'{CLASS_NAMES[i]} (AUC={pr_auc:.3f})')
ax2.set_xlabel('Recall', fontsize=10)
ax2.set_ylabel('Precision', fontsize=10)
ax2.legend(loc='lower left', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_title('Precision-Recall Curves', fontsize=11)

plt.tight_layout()
out_path = os.path.join(RESULTS_DIR, 'roc_curves_final.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nSaved ROC curves to {out_path}")

roc_data = {
    'macro_auc': float(macro_auc),
    'per_class_auc': {CLASS_NAMES[i]: float(auc(
        roc_curve(l_bin[:,i], pr[:,i])[0],
        roc_curve(l_bin[:,i], pr[:,i])[1]
    )) for i in present_classes if l_bin[:,i].sum() > 0}
}
with open(os.path.join(RESULTS_DIR, 'roc_data_final.json'), 'w') as f:
    json.dump(roc_data, f, indent=2)
print(f"Macro AUC: {macro_auc:.4f}")
print("Done.")
