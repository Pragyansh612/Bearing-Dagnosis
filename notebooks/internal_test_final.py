import os, sys, json
import numpy as np
import torch
import joblib
import scipy.io as sio
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, confusion_matrix, roc_auc_score)
from sklearn.preprocessing import label_binarize
from sklearn.metrics import matthews_corrcoef, cohen_kappa_score
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

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
RESULTS_DIR  = os.path.expanduser('~/bearing_diagnosis/results')
DEVICE       = torch.device('cuda')
CLASS_NAMES  = ['Healthy', 'Inner Race Fault', 'Ball Fault', 'Outer Race Fault']
FOLDERS      = ['1','2','3','4','5','6','7']
WINDOW_SIZE  = 2048
STEP         = 1024

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

def load_mat(fpath):
    if not os.path.exists(fpath): return None
    mat = sio.loadmat(fpath, simplify_cells=True)
    fault_type   = int(mat.get('faultType', 0))
    fault_origin = str(mat.get('faultOrigin', 'DS'))
    src = get_signal_source(mat, fault_origin)
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
    measurements = []
    if raw_data.ndim==2 and raw_data.dtype.kind in ('f','i','u'):
        for i in range(raw_data.shape[0]):
            measurements.append((raw_data[i].astype(np.float32),
                float(rpm_vals[i]) if i<len(rpm_vals) else 1200.0,
                float(fs_vals[i])  if i<len(fs_vals)  else 5120.0))
    else:
        flat = raw_data.flatten()
        for i in range(len(flat)):
            try:
                sig = extract_1d_signal(flat[i])
                measurements.append((sig,
                    float(rpm_vals[i]) if i<len(rpm_vals) else 1200.0,
                    float(fs_vals[i])  if i<len(fs_vals)  else 5120.0))
            except: continue
    return measurements, fault_type, fault_freq

print("Loading model and scaler...")
model  = OrderModel().to(DEVICE)
ckpt   = torch.load(os.path.join(MODELS_DIR, 'best_model_final_best.pth'))
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
scaler = joblib.load(os.path.join(MODELS_DIR, 'physics_scaler_final_best.pkl'))
print(f"  epoch={ckpt['epoch']}, val_f1={ckpt['val_f1']:.4f}")

print("\nLoading internal test data (folders 1-7, test split)...")
all_sigs, all_labs, all_rpms, all_fss, all_ffs = [], [], [], [], []
for folder in FOLDERS:
    result = load_mat(os.path.join(DATASET_PATH, folder, 'test.mat'))
    if result is None: continue
    measurements, fault_type, fault_freq = result
    n = len(measurements)
    start = int(n * 0.85)
    for sig, rpm, fs in measurements[start:]:
        if len(sig) < WINDOW_SIZE: continue
        for s in range(0, len(sig)-WINDOW_SIZE+1, STEP):
            all_sigs.append(sig[s:s+WINDOW_SIZE])
            all_labs.append(fault_type)
            all_rpms.append(rpm)
            all_fss.append(fs)
            all_ffs.append(fault_freq)

print(f"  Loaded {len(all_sigs)} windows")
u, c = np.unique(all_labs, return_counts=True)
for uu, cc in zip(u, c):
    print(f"    {CLASS_NAMES[int(uu)]}: {cc}")

print("\nRunning inference...")
all_preds, all_probs = [], []
BATCH = 64
model.eval()
for i in range(0, len(all_sigs), BATCH):
    batch_sigs = all_sigs[i:i+BATCH]
    batch_rpms = all_rpms[i:i+BATCH]
    batch_fss  = all_fss[i:i+BATCH]
    batch_ffs  = all_ffs[i:i+BATCH]
    bt, bf, bp = [], [], []
    for j in range(len(batch_sigs)):
        sig  = batch_sigs[j]
        rpm  = float(batch_rpms[j])
        fs   = float(batch_fss[j])
        ff   = batch_ffs[j]
        phys = compute_order_physics_features(sig, rpm, fs, ff)
        ps   = scaler.transform(phys.reshape(1,-1)).astype(np.float32).flatten()
        sn   = normalize_signal(sig)
        bt.append(sn)
        bf.append(compute_order_stft(sn, rpm, fs))
        bp.append(ps)
    x_t = torch.from_numpy(np.array(bt)).float().to(DEVICE)
    x_f = torch.from_numpy(np.array(bf)).float().to(DEVICE)
    x_p = torch.from_numpy(np.array(bp)).float().to(DEVICE)
    with torch.no_grad():
        logits, _ = model(x_t, x_f, x_p)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
    all_preds.extend(logits.argmax(-1).cpu().numpy().tolist())
    all_probs.extend(probs.tolist())
    if i % 500 == 0: print(f"  {i}/{len(all_sigs)}")

p  = np.array(all_preds)
l  = np.array(all_labs)
pr = np.array(all_probs)

acc   = accuracy_score(l, p)
f1    = f1_score(l, p, average='macro', zero_division=0)
prec  = precision_score(l, p, average='macro', zero_division=0)
rec   = recall_score(l, p, average='macro', zero_division=0)
mcc   = matthews_corrcoef(l, p)
kappa = cohen_kappa_score(l, p)
f1_per   = f1_score(l, p, average=None, zero_division=0, labels=[0,1,2,3])
prec_per = precision_score(l, p, average=None, zero_division=0, labels=[0,1,2,3])
rec_per  = recall_score(l, p, average=None, zero_division=0, labels=[0,1,2,3])
cm = confusion_matrix(l, p, labels=[0,1,2,3])
try:
    l_bin   = label_binarize(l, classes=[0,1,2,3])
    roc_auc = roc_auc_score(l_bin, pr, average='macro', multi_class='ovr')
except: roc_auc = 0.0

print(f"\n{'='*70}")
print(f"INTERNAL TEST RESULTS — FOLDERS 1-7 (known machines)")
print(f"{'='*70}")
print(f"  Accuracy:  {acc:.4f} ({acc:.1%})")
print(f"  Macro F1:  {f1:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall:    {rec:.4f}")
print(f"  ROC AUC:   {roc_auc:.4f}")
print(f"  MCC:       {mcc:.4f}")
print(f"  Kappa:     {kappa:.4f}")
print(f"\n  {'Class':<20} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Sup':>8}")
print(f"  {'-'*48}")
for i, cls in enumerate(CLASS_NAMES):
    sup = int((l==i).sum())
    if sup > 0:
        print(f"  {cls:<20} {prec_per[i]:>6.4f} {rec_per[i]:>6.4f} {f1_per[i]:>6.4f} {sup:>8}")
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
    'dataset': 'internal_folders_1_to_7',
    'accuracy': float(acc), 'f1_macro': float(f1),
    'precision': float(prec), 'recall': float(rec),
    'roc_auc': float(roc_auc), 'mcc': float(mcc), 'kappa': float(kappa),
    'confusion_matrix': cm.tolist(),
    'per_class': {CLASS_NAMES[i]: {
        'precision': float(prec_per[i]),
        'recall': float(rec_per[i]),
        'f1': float(f1_per[i]),
        'support': int((l==i).sum()),
    } for i in range(4)},
    'hidden_test': {
        'accuracy': 0.6367, 'f1_macro': 0.4247,
        'note': 'from hidden_v5_results.json'
    }
}
with open(os.path.join(RESULTS_DIR, 'internal_test_final.json'), 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to results/internal_test_final.json")
