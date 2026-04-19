import os, sys, json
import numpy as np
import torch
import joblib
import scipy.io as sio
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.expanduser('~/bearing_diagnosis'))

from src.models.time_branch import TimeDomainBranch
from src.models.freq_branch import FrequencyBranch
from src.models.physics_branch import PhysicsBranch
from src.models.fusion import AttentionFusion
from src.models.autoencoder import BearingAutoencoder
from src.preprocessing.signal_processor_v2 import (
    compute_order_stft, compute_order_physics_features, normalize_signal
)

DATASET_PATH         = os.path.expanduser('~') + '/SCA bearing dataset/data'
MODELS_DIR           = os.path.expanduser('~/bearing_diagnosis/models')
RESULTS_DIR          = os.path.expanduser('~/bearing_diagnosis/results')
DEVICE               = torch.device('cuda')
CLASS_NAMES          = ['Healthy', 'Inner Race Fault', 'Ball Fault', 'Outer Race Fault']
HIDDEN_FOLDERS       = ['8', '9', '10']
WINDOW_SIZE          = 2048
N_SAMPLES            = 100
N_MC                 = 50
UNCERTAINTY_THRESHOLD    = 0.15
PHYSICS_ENERGY_THRESHOLD = 0.01
AE_SIGMA_THRESHOLD       = 8.0
HEALTHY_MARGIN_THRESHOLD = 0.08
BPFO_ENERGY_THRESHOLD    = 0.015
KURTOSIS_HEALTHY_MAX     = 3.2
CREST_HEALTHY_MAX        = 4.5
BPFO_INDICES = list(range(8,12)) + [16,17]
FAULT_ENERGY_INDICES = {
    1: list(range(12,16)) + [18,19],
    2: list(range(4,8)),
    3: list(range(8,12)) + [16,17],
}

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

model = OrderModel().to(DEVICE)
ckpt  = torch.load(os.path.join(MODELS_DIR, 'best_model_final_best.pth'))
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

def compute_signal_stats(signal):
    rms   = float(np.sqrt(np.mean(signal**2)))
    peak  = float(np.abs(signal).max())
    crest = peak / (rms + 1e-8)
    std   = signal.std() + 1e-8
    kurt  = float(np.mean(((signal - signal.mean()) / std) ** 4))
    return rms, crest, kurt

def check_physics(pred_class, phys_raw, rms):
    if pred_class == 0: return True, 0.0
    indices = FAULT_ENERGY_INDICES.get(pred_class, [])
    if not indices: return True, 0.0
    fault_energy = float(phys_raw[indices].mean())
    relative     = fault_energy / (rms + 1e-8)
    return relative > PHYSICS_ENERGY_THRESHOLD, fault_energy

def infer(signal, rpm, fs, ff,
          use_emb_norm=True, use_margin=True,
          use_bpfo=True, use_health_check=True):
    rms, crest, kurt = compute_signal_stats(signal)
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

    if use_emb_norm:
        emb = emb_np - emb_np.mean()
        emb = emb / (emb.std() + 1e-6)
        emb = emb / (np.linalg.norm(emb) + 1e-8)
    else:
        emb = emb_np / (np.linalg.norm(emb_np) + 1e-8)

    sims        = centers @ emb
    proto_class = int(np.argmax(sims))

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
    mc_preds  = np.array(mc_preds)
    mean_pred = mc_preds.mean(axis=0)

    pred_class = proto_class

    if use_margin and pred_class == 0:
        sorted_idx   = np.argsort(sims)
        second_class = int(sorted_idx[-2])
        margin       = float(sims[pred_class] - sims[second_class])
        if margin < HEALTHY_MARGIN_THRESHOLD:
            pred_class = second_class

    if use_bpfo and (pred_class == 0 or pred_class == 1):
        bpfo_energy = float(phys_raw[BPFO_INDICES].mean()) / (rms + 1e-8)
        if bpfo_energy > BPFO_ENERGY_THRESHOLD:
            pred_class = 3

    physics_ok, _ = check_physics(pred_class, phys_raw, rms)
    if use_health_check and pred_class != 0 and not physics_ok:
        if kurt < KURTOSIS_HEALTHY_MAX and crest < CREST_HEALTHY_MAX:
            pred_class = 0

    return pred_class, mean_pred

def run_config(name, use_emb_norm, use_margin, use_bpfo, use_health_check):
    all_true, all_pred = [], []
    for folder in HIDDEN_FOLDERS:
        for mat_file in ['train.mat', 'test.mat']:
            samples = load_samples(folder, mat_file, N_SAMPLES)
            if not samples: continue
            for sig, ft, rpm, fs, ff in samples:
                pc, _ = infer(sig, rpm, fs, ff,
                              use_emb_norm=use_emb_norm,
                              use_margin=use_margin,
                              use_bpfo=use_bpfo,
                              use_health_check=use_health_check)
                all_true.append(ft)
                all_pred.append(pc)
    acc = accuracy_score(all_true, all_pred)
    f1  = f1_score(all_true, all_pred, average='macro', zero_division=0)
    return acc, f1

configs = [
    ('V5 FULL (all components)',         True,  True,  True,  True),
    ('No embedding normalization',        False, True,  True,  True),
    ('No margin override',                True,  False, True,  True),
    ('No BPFO physics override',          True,  True,  False, True),
    ('No health check',                   True,  True,  True,  False),
    ('Prototype only (no overrides)',     True,  False, False, False),
    ('Raw softmax only (no prototype)',   False, False, False, False),
]

print(f"\n{'='*70}")
print(f"ABLATION STUDY — HIDDEN FOLDERS 8, 9, 10")
print(f"{'='*70}")
print(f"\n  {'Configuration':<40} {'Accuracy':>10} {'Macro F1':>10} {'Delta Acc':>10}")
print(f"  {'-'*72}")

results_ablation = []
baseline_acc = None

for name, en, mg, bp, hc in configs:
    print(f"  Running: {name}...", flush=True)
    acc, f1 = run_config(name, en, mg, bp, hc)
    if baseline_acc is None:
        baseline_acc = acc
        delta = 0.0
    else:
        delta = acc - baseline_acc
    results_ablation.append({
        'config': name,
        'accuracy': float(acc),
        'f1_macro': float(f1),
        'delta_vs_full': float(delta),
    })
    delta_str = f'{delta:+.4f}' if delta != 0.0 else '(baseline)'
    print(f"  {name:<40} {acc:>10.4f} {f1:>10.4f} {delta_str:>10}")

print(f"\n{'='*70}")
print(f"COMPONENT CONTRIBUTION SUMMARY")
print(f"{'='*70}")
for r in results_ablation[1:]:
    drop = r['delta_vs_full']
    print(f"  Removing {r['config']:<35}: {drop:+.4f} accuracy")

with open(os.path.join(RESULTS_DIR, 'ablation_final.json'), 'w') as f:
    json.dump(results_ablation, f, indent=2)
print(f"\nSaved to results/ablation_final.json")
