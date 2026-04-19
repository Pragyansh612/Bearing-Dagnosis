import os, sys
import numpy as np
import torch
import joblib
import scipy.io as sio

sys.path.insert(0, os.path.expanduser('~/bearing_diagnosis'))

from src.models.time_branch import TimeDomainBranch
from src.models.freq_branch import FrequencyBranch
from src.models.physics_branch import PhysicsBranch
from src.models.fusion import AttentionFusion
from src.preprocessing.signal_processor_v2 import (
    compute_order_stft, compute_order_physics_features, normalize_signal
)

DATASET_PATH = os.path.expanduser('~') + '/SCA bearing dataset/data'
MODELS_DIR   = os.path.expanduser('~/bearing_diagnosis/models')
FOLDERS      = ['1','2','3','4','5','6','7']
WINDOW_SIZE  = 2048
DEVICE       = torch.device('cuda')
CLASS_NAMES  = ['Healthy', 'Inner Race Fault', 'Ball Fault', 'Outer Race Fault']

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

def load_samples_from_file(folder, mat_file, n_per_file=200):
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
        if len(sig) < WINDOW_SIZE: continue
        mid = len(sig)//2 - WINDOW_SIZE//2
        w   = sig[mid:mid+WINDOW_SIZE]
        samples.append((w, fault_type, rpm, fs, fault_freq))
        if len(samples) >= n_per_file: break
    return samples

print("Loading model...")
model = OrderModel().to(DEVICE)
ckpt  = torch.load(os.path.join(MODELS_DIR, 'best_model_final_best.pth'))
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
scaler = joblib.load(os.path.join(MODELS_DIR, 'physics_scaler_final_best.pkl'))
print(f"  Loaded epoch={ckpt['epoch']}, val_f1={ckpt['val_f1']:.4f}")

print("\nCollecting embeddings from folders 1-7 (train.mat AND test.mat)...")
class_embeddings = {0: [], 1: [], 2: [], 3: []}
class_counts     = {0: 0,  1: 0,  2: 0,  3: 0}

for folder in FOLDERS:
    for mat_file in ['train.mat', 'test.mat']:
        samples = load_samples_from_file(folder, mat_file, n_per_file=200)
        if not samples: continue
        fault_type = samples[0][1]
        print(f"  Folder {folder}/{mat_file}: {len(samples)} samples | class={CLASS_NAMES[fault_type]}")

        batch_t, batch_f, batch_p = [], [], []
        for sig, ft, rpm, fs, ff in samples:
            phys    = compute_order_physics_features(sig, rpm, fs, ff)
            phys_sc = scaler.transform(phys.reshape(1,-1)).astype(np.float32).flatten()
            sig_n   = normalize_signal(sig)
            ostft   = compute_order_stft(sig_n, rpm, fs)
            batch_t.append(sig_n)
            batch_f.append(ostft)
            batch_p.append(phys_sc)

        x_t = torch.from_numpy(np.array(batch_t)).float().to(DEVICE)
        x_f = torch.from_numpy(np.array(batch_f)).float().to(DEVICE)
        x_p = torch.from_numpy(np.array(batch_p)).float().to(DEVICE)

        with torch.no_grad():
            embs = model.get_fused(x_t, x_f, x_p).cpu().numpy()

        class_embeddings[fault_type].extend(embs)
        class_counts[fault_type] += len(embs)

print("\nComputing class centers...")
centers = np.zeros((4, 256), dtype=np.float32)
for c in range(4):
    if len(class_embeddings[c]) == 0:
        print(f"  WARNING: No embeddings for class {CLASS_NAMES[c]}")
        continue
    embs_arr   = np.array(class_embeddings[c], dtype=np.float32)
    center     = embs_arr.mean(axis=0)
    center     = center / (np.linalg.norm(center) + 1e-8)
    centers[c] = center
    print(f"  {CLASS_NAMES[c]:<20}: {class_counts[c]:4d} embeddings | norm={np.linalg.norm(centers[c]):.4f}")

counts_arr = np.array([class_counts[c] for c in range(4)], dtype=np.int32)
np.save(os.path.join(MODELS_DIR, 'class_centers.npy'), centers)
np.save(os.path.join(MODELS_DIR, 'class_counts.npy'),  counts_arr)
print(f"\nSaved to {MODELS_DIR}")
print("Done.")
