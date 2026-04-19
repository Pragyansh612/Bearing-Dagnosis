import os, sys, json
import numpy as np
import torch
import joblib
import scipy.io as sio
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, confusion_matrix)
from sklearn.metrics import matthews_corrcoef, cohen_kappa_score

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

FAULT_ENERGY_INDICES = {
    1: list(range(12,16)) + [18,19],
    2: list(range(4,8)),
    3: list(range(8,12)) + [16,17],
}
BPFO_INDICES = list(range(8,12)) + [16,17]

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

print("Loading models...")
model = OrderModel().to(DEVICE)
ckpt  = torch.load(os.path.join(MODELS_DIR, 'best_model_final_best.pth'))
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
scaler = joblib.load(os.path.join(MODELS_DIR, 'physics_scaler_final_best.pkl'))

ae = BearingAutoencoder(input_dim=256, latent_dim=32).to(DEVICE)
ae_ckpt = torch.load(os.path.join(MODELS_DIR, 'ae_final_best.pth'))
ae.load_state_dict(ae_ckpt['model_state_dict'])
ae.mean_error = ae_ckpt['mean_error']
ae.std_error  = ae_ckpt['std_error']
ae.threshold  = ae_ckpt['threshold']
ae.eval()

centers = np.load(os.path.join(MODELS_DIR, 'class_centers.npy'))
counts  = np.load(os.path.join(MODELS_DIR, 'class_counts.npy'))
print(f"  Model: epoch={ckpt['epoch']}, val_f1={ckpt['val_f1']:.4f}")
print(f"  Class centers: {centers.shape}")
for i, cls in enumerate(CLASS_NAMES):
    print(f"    {cls:<20}: {counts[i]} embeddings")

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

def normalize_embedding(emb):
    # Layer 1: mean-center + std-normalize (removes machine-specific bias)
    emb = emb - emb.mean()
    emb = emb / (emb.std() + 1e-6)
    # Layer 2: L2 normalize (unit sphere for cosine)
    emb = emb / (np.linalg.norm(emb) + 1e-8)
    return emb

def cosine_classify(embedding):
    emb_norm = normalize_embedding(embedding)
    sims     = centers @ emb_norm
    return sims

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

def enable_dropout(m):
    if isinstance(m, torch.nn.Dropout): m.train()

def infer(signal, rpm, fs, ff):
    rms, crest, kurt = compute_signal_stats(signal)
    phys_raw = compute_order_physics_features(signal, rpm, fs, ff)
    phys_sc  = scaler.transform(phys_raw.reshape(1,-1)).astype(np.float32)
    sig_norm = normalize_signal(signal)
    x_time   = torch.from_numpy(sig_norm).unsqueeze(0).to(DEVICE)
    x_freq   = torch.from_numpy(compute_order_stft(sig_norm, rpm, fs)).unsqueeze(0).to(DEVICE)
    x_phys   = torch.from_numpy(phys_sc).to(DEVICE)

    # Get fused embedding + AE error (AE = info only)
    model.eval()
    with torch.no_grad():
        fused = model.get_fused(x_time, x_freq, x_phys)
    ae.eval()
    with torch.no_grad():
        ae_error = float(ae.normalized_error(fused).item())

    emb_np = fused.cpu().numpy()[0]

    # LAYER 1 — Prototype with normalized embedding
    sims        = cosine_classify(emb_np)
    proto_class = int(np.argmax(sims))
    proto_conf  = float(sims[proto_class])

    # MC Dropout — uncertainty only
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
    mc_preds      = np.array(mc_preds)
    mean_pred     = mc_preds.mean(axis=0)
    softmax_class = int(mean_pred.argmax())
    uncertainty   = float(mc_preds.std(axis=0)[proto_class])

    pred_class = proto_class
    override_reason = None

    # LAYER 2 — Margin override: if healthy barely wins, pick second
    if pred_class == 0:
        sorted_idx = np.argsort(sims)
        second_class = int(sorted_idx[-2])
        margin = float(sims[pred_class] - sims[second_class])
        if margin < HEALTHY_MARGIN_THRESHOLD:
            pred_class = second_class
            override_reason = f'MARGIN({margin:.3f})'

    # LAYER 3 — Physics BPFO override: if BPFO energy strong, force outer race
    if pred_class == 0 or pred_class == 1:
        bpfo_energy = float(phys_raw[BPFO_INDICES].mean()) / (rms + 1e-8)
        if bpfo_energy > BPFO_ENERGY_THRESHOLD:
            pred_class = 3
            override_reason = f'BPFO({bpfo_energy:.4f})'

    # LAYER 4 — Conservative health check (only when fault predicted + physics mismatch)
    # Gate: must have physics mismatch AND signal looks statistically healthy
    physics_ok, fault_energy = check_physics(pred_class, phys_raw, rms)
    if pred_class != 0 and not physics_ok:
        if kurt < KURTOSIS_HEALTHY_MAX and crest < CREST_HEALTHY_MAX:
            pred_class = 0
            override_reason = f'HEALTH_CHECK(k={kurt:.2f},c={crest:.2f})'

    # Recompute physics_ok for final pred_class
    physics_ok, fault_energy = check_physics(pred_class, phys_raw, rms)

    # Decision label (AE = info only)
    ae_flag = ae_error > AE_SIGMA_THRESHOLD
    if uncertainty > UNCERTAINTY_THRESHOLD:
        decision = 'UNCERTAIN'
    elif pred_class != 0 and not physics_ok:
        decision = 'PHYSICS_MISMATCH'
    elif ae_flag:
        decision = 'LOW_CONFIDENCE'
    else:
        decision = 'CONFIDENT'

    return {
        'pred_class':     pred_class,
        'pred_name':      CLASS_NAMES[pred_class],
        'confidence':     proto_conf,
        'uncertainty':    uncertainty,
        'ae_error':       ae_error,
        'ae_flag':        ae_flag,
        'decision':       decision,
        'mean_pred':      mean_pred,
        'softmax_class':  softmax_class,
        'proto_class':    proto_class,
        'override':       override_reason,
        'kurt':           kurt,
        'crest':          crest,
    }

print(f"\n{'='*70}")
print(f"HIDDEN FOLDER TEST — V5 — EMB NORM + MARGIN + PHYSICS OVERRIDES")
print(f"{'='*70}")

all_true, all_pred, all_probs, all_decisions = [], [], [], []

for folder in HIDDEN_FOLDERS:
    for mat_file in ['train.mat', 'test.mat']:
        samples = load_samples(folder, mat_file, N_SAMPLES)
        if not samples: continue
        true_class = samples[0][1]
        results_f  = []

        for sig, ft, rpm, fs, ff in samples:
            r = infer(sig, rpm, fs, ff)
            results_f.append(r)
            all_true.append(ft)
            all_pred.append(r['pred_class'])
            all_probs.append(r['mean_pred'])
            all_decisions.append(r['decision'])

        preds        = [r['pred_class'] for r in results_f]
        correct      = sum(1 for p in preds if p == true_class)
        ae_mean      = np.mean([r['ae_error'] for r in results_f])
        proto_agree  = sum(1 for r in results_f if r['proto_class'] == r['softmax_class'])
        overrides    = [r['override'] for r in results_f if r['override'] is not None]

        print(f"\n  Folder {folder}/{mat_file} — True: {CLASS_NAMES[true_class]}")
        print(f"  Correct: {correct}/{len(samples)} ({correct/len(samples):.1%}) | AE: {ae_mean:.1f}σ | Proto=Softmax: {proto_agree}/{len(samples)}")
        if overrides:
            margin_cnt = sum(1 for o in overrides if o.startswith('MARGIN'))
            bpfo_cnt   = sum(1 for o in overrides if o.startswith('BPFO'))
            health_cnt = sum(1 for o in overrides if o.startswith('HEALTH'))
            if margin_cnt: print(f"  Overrides — MARGIN: {margin_cnt}")
            if bpfo_cnt:   print(f"  Overrides — BPFO: {bpfo_cnt}")
            if health_cnt: print(f"  Overrides — HEALTH_CHECK: {health_cnt}")
        decisions = [r['decision'] for r in results_f]
        for d in ['CONFIDENT','LOW_CONFIDENCE','UNCERTAIN','PHYSICS_MISMATCH']:
            cnt = decisions.count(d)
            if cnt > 0: print(f"  {d}: {cnt}")
        pred_arr = np.array(preds)
        for i, cls in enumerate(CLASS_NAMES):
            cnt = (pred_arr==i).sum()
            if cnt > 0: print(f"  → {cls}: {cnt} ({cnt/len(preds):.1%})")

print(f"\n\n{'='*70}")
print(f"OVERALL — V5 — HIDDEN FOLDERS 8, 9, 10")
print(f"{'='*70}")

p = np.array(all_pred)
l = np.array(all_true)

acc   = accuracy_score(l, p)
f1    = f1_score(l, p, average='macro', zero_division=0)
prec  = precision_score(l, p, average='macro', zero_division=0)
rec   = recall_score(l, p, average='macro', zero_division=0)
mcc   = matthews_corrcoef(l, p)
kappa = cohen_kappa_score(l, p)

print(f"\n  Total: {len(l)}")
print(f"  Accuracy:  {acc:.4f} ({acc:.1%})")
print(f"  Macro F1:  {f1:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall:    {rec:.4f}")
print(f"  MCC:       {mcc:.4f}")
print(f"  Kappa:     {kappa:.4f}")

decisions_arr = np.array(all_decisions)
print(f"\n  Decisions:")
for d in ['CONFIDENT','LOW_CONFIDENCE','UNCERTAIN','PHYSICS_MISMATCH']:
    cnt = (decisions_arr==d).sum()
    if cnt > 0: print(f"  {d:<20}: {cnt:4d} ({cnt/len(decisions_arr):.1%})")

f1_per   = f1_score(l, p, average=None, zero_division=0, labels=[0,1,2,3])
prec_per = precision_score(l, p, average=None, zero_division=0, labels=[0,1,2,3])
rec_per  = recall_score(l, p, average=None, zero_division=0, labels=[0,1,2,3])
cm       = confusion_matrix(l, p, labels=[0,1,2,3])

print(f"\n  Per-class:")
print(f"  {'Class':<20} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Sup':>8}")
for i, cls in enumerate(CLASS_NAMES):
    sup = int((l==i).sum())
    if sup > 0:
        print(f"  {cls:<20} {prec_per[i]:>6.4f} {rec_per[i]:>6.4f} "
              f"{f1_per[i]:>6.4f} {sup:>8}")

print(f"\n  Confusion Matrix:")
header = f"  {'':20}"
for cls in CLASS_NAMES: header += f"  {cls[:8]:>10}"
print(header)
for i, cls in enumerate(CLASS_NAMES):
    if (l==i).sum() > 0:
        row = f"  {cls:<20}"
        for j in range(4): row += f"  {cm[i,j]:>10}"
        print(row)

results = {
    'version': 'v5_embnorm_margin_physics',
    'hidden_folders': HIDDEN_FOLDERS,
    'thresholds': {
        'ae_sigma': AE_SIGMA_THRESHOLD,
        'healthy_margin': HEALTHY_MARGIN_THRESHOLD,
        'bpfo_energy': BPFO_ENERGY_THRESHOLD,
        'kurtosis_healthy_max': KURTOSIS_HEALTHY_MAX,
        'crest_healthy_max': CREST_HEALTHY_MAX,
    },
    'accuracy': float(acc), 'f1_macro': float(f1),
    'precision': float(prec), 'recall': float(rec),
    'mcc': float(mcc), 'kappa': float(kappa),
    'confusion_matrix': cm.tolist(),
    'per_class': {CLASS_NAMES[i]: {
        'precision': float(prec_per[i]),
        'recall': float(rec_per[i]),
        'f1': float(f1_per[i]),
        'support': int((l==i).sum()),
    } for i in range(4)},
}
with open(os.path.join(RESULTS_DIR, 'hidden_v5_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to results/hidden_v5_results.json")
