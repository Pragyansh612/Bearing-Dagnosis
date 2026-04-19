import os, sys, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, confusion_matrix, roc_auc_score)
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import matthews_corrcoef, cohen_kappa_score
import joblib
import scipy.io as sio

sys.path.insert(0, os.path.expanduser('~/bearing_diagnosis'))

from src.models.time_branch import TimeDomainBranch
from src.models.freq_branch import FrequencyBranch
from src.models.physics_branch import PhysicsBranch
from src.models.fusion import AttentionFusion
from src.models.autoencoder import BearingAutoencoder
from src.training.physics_loss import combined_loss
from src.preprocessing.bearing_dataset import BearingDataset
from src.preprocessing.signal_processor_v2 import (
    compute_order_stft, compute_order_physics_features, normalize_signal
)
from src.preprocessing.augmentation import augment_signal

DATASET_PATH = os.path.expanduser('~') + '/SCA bearing dataset/data'
CLASS_NAMES  = ['Healthy', 'Inner Race Fault', 'Ball Fault', 'Outer Race Fault']

CLASSIFIER_FOLDERS = ['1','2','3','4','5','6','7']
AE_FOLDERS         = ['1','2','3','4','5','6','7']
HIDDEN_FOLDERS     = ['8','9','10']

WINDOW_SIZE  = 2048
STEP         = 1024
BATCH_SIZE   = 128  # 32 per class × 4 classes
EPOCHS       = 100
LR           = 1e-3
WEIGHT_DECAY = 1e-4
LAMBDA_PHYS  = 0.5
LAMBDA_TRIP  = 0.1
LABEL_SMOOTH = 0.1
PATIENCE     = 15
MARGIN       = 1.0
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_DIR  = os.path.expanduser('~/bearing_diagnosis/results')
MODELS_DIR   = os.path.expanduser('~/bearing_diagnosis/models')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

# ── Cross-folder balanced sampler ──────────────────────────────────────────

class CrossFolderBalancedSampler(Sampler):
    """
    Each batch has equal samples per class (32 per class = 128 total).
    Samples are drawn from DIFFERENT folders within each class.
    This enforces cross-folder diversity every batch.
    """
    def __init__(self, labels, folders, n_per_class=32, n_classes=4):
        self.labels      = np.array(labels)
        self.folders     = np.array(folders)
        self.n_per_class = n_per_class
        self.n_classes   = n_classes

        # Build index lookup: class → {folder → [indices]}
        self.class_folder_indices = {}
        for c in range(n_classes):
            self.class_folder_indices[c] = {}
            class_mask = self.labels == c
            class_folders = np.unique(self.folders[class_mask])
            for f in class_folders:
                mask = class_mask & (self.folders == f)
                self.class_folder_indices[c][f] = np.where(mask)[0]

        # Estimate number of batches
        min_class_size = min(
            sum(len(v) for v in self.class_folder_indices[c].values())
            for c in range(n_classes)
            if self.class_folder_indices[c]
        )
        self.n_batches = max(1, min_class_size // n_per_class)

    def __iter__(self):
        all_indices = []
        for _ in range(self.n_batches):
            batch = []
            for c in range(self.n_classes):
                if not self.class_folder_indices[c]:
                    continue
                # Get available folders for this class
                avail_folders = list(self.class_folder_indices[c].keys())
                if len(avail_folders) == 1:
                    indices = list(self.class_folder_indices[c].values())[0]
                    chosen = np.random.choice(indices, size=self.n_per_class, replace=True)
                    batch.extend(chosen.tolist())
                    continue

                # Sample from multiple folders — cross-folder diversity
                selected = []
                remaining = self.n_per_class
                np.random.shuffle(avail_folders)
                for f in avail_folders:
                    indices = self.class_folder_indices[c][f]
                    n_take = min(remaining, max(1, remaining // len(avail_folders)))
                    chosen = np.random.choice(indices, size=min(n_take, len(indices)),
                                               replace=True)
                    selected.extend(chosen.tolist())
                    remaining -= len(chosen)
                    if remaining <= 0:
                        break

                # Fill remaining if needed
                if remaining > 0:
                    all_class_idx = np.concatenate(
                        list(self.class_folder_indices[c].values())
                    )
                    extra = np.random.choice(all_class_idx, size=remaining, replace=True)
                    selected.extend(extra.tolist())

                batch.extend(selected[:self.n_per_class])

            np.random.shuffle(batch)
            all_indices.extend(batch)
        return iter(all_indices)

    def __len__(self):
        return self.n_batches * self.n_classes * self.n_per_class

# ── Cross-folder semi-hard triplet loss ────────────────────────────────────

class CrossFolderTripletLoss(nn.Module):
    """
    Semi-hard triplet loss with cross-folder enforcement.
    
    For each anchor:
    - Positive: same class, DIFFERENT folder
    - Negative: different class (semi-hard: d(a,n) > d(a,p) but < d(a,p) + margin)
    
    Semi-hard negatives are more stable than hardest negatives.
    """
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, features, labels, folders):
        device    = features.device
        labels_np = labels.cpu().numpy()
        folders_np = folders.cpu().numpy()
        n = len(labels_np)

        # Compute all pairwise distances
        dist_matrix = torch.cdist(features, features, p=2)

        triplet_loss = torch.tensor(0.0, device=device)
        n_triplets   = 0

        for i in range(n):
            anchor_label  = labels_np[i]
            anchor_folder = folders_np[i]

            # Find cross-folder positives (same class, DIFFERENT folder)
            pos_mask = (labels_np == anchor_label) & \
                       (folders_np != anchor_folder)
            pos_indices = np.where(pos_mask)[0]
            if len(pos_indices) == 0:
                # Fallback to same-folder positive if no cross-folder available
                pos_mask = (labels_np == anchor_label) & \
                           (np.arange(n) != i)
                pos_indices = np.where(pos_mask)[0]
            if len(pos_indices) == 0:
                continue

            # Pick hardest positive (farthest same-class sample)
            pos_dists = dist_matrix[i][torch.tensor(pos_indices, device=device)]
            p_idx     = pos_indices[pos_dists.argmax().item()]
            d_ap      = dist_matrix[i][p_idx]

            # Find semi-hard negatives
            # d(a,n) > d(a,p) AND d(a,n) < d(a,p) + margin
            neg_mask = labels_np != anchor_label
            neg_indices = np.where(neg_mask)[0]
            if len(neg_indices) == 0:
                continue

            neg_dists = dist_matrix[i][torch.tensor(neg_indices, device=device)]
            # Semi-hard: farther than positive but within margin
            semi_hard = (neg_dists > d_ap) & (neg_dists < d_ap + self.margin)

            if semi_hard.sum() > 0:
                # Use semi-hard negatives
                semi_hard_idx = neg_indices[semi_hard.cpu().numpy()]
                n_idx = semi_hard_idx[np.random.randint(len(semi_hard_idx))]
            else:
                # Fall back to hardest negative
                n_idx = neg_indices[neg_dists.argmax().item()]

            d_an = dist_matrix[i][n_idx]
            loss = F.relu(d_ap - d_an + self.margin)
            triplet_loss += loss
            n_triplets   += 1

        if n_triplets > 0:
            triplet_loss = triplet_loss / n_triplets
        return triplet_loss

# ── Helpers ────────────────────────────────────────────────────────────────

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

def to_windows(measurements, fault_type, fault_freq, folder):
    windows = []
    for sig, rpm, fs in measurements:
        if rpm<=0: rpm=1200.0
        if fs<=0:  fs=5120.0
        if len(sig)<WINDOW_SIZE: continue
        for start in range(0, len(sig)-WINDOW_SIZE+1, STEP):
            w = sig[start:start+WINDOW_SIZE]
            windows.append((w, fault_type, rpm, fs, fault_freq, folder))
    return windows

def load_classifier_data():
    train_data, val_data, test_data = [], [], []
    for folder in CLASSIFIER_FOLDERS:
        folder_path = os.path.join(DATASET_PATH, folder)
        result = load_mat(os.path.join(folder_path, 'train.mat'))
        if result:
            measurements, fault_type, fault_freq = result
            n = len(measurements)
            n_tr = int(n*0.85); n_va = int(n*0.075)
            train_data.extend(to_windows(measurements[:n_tr], fault_type, fault_freq, folder))
            val_data.extend(to_windows(measurements[n_tr:n_tr+n_va], fault_type, fault_freq, folder))
            test_data.extend(to_windows(measurements[n_tr+n_va:], fault_type, fault_freq, folder))
        result = load_mat(os.path.join(folder_path, 'test.mat'))
        if result:
            measurements, fault_type, fault_freq = result
            n = len(measurements)
            n_tr = int(n*0.70); n_va = int(n*0.15)
            train_data.extend(to_windows(measurements[:n_tr], fault_type, fault_freq, folder))
            val_data.extend(to_windows(measurements[n_tr:n_tr+n_va], fault_type, fault_freq, folder))
            test_data.extend(to_windows(measurements[n_tr+n_va:], fault_type, fault_freq, folder))

    def to_arrays(data):
        return (
            np.array([d[0] for d in data], dtype=np.float32),
            np.array([d[1] for d in data], dtype=np.int64),
            np.array([d[2] for d in data], dtype=np.float32),
            np.array([d[3] for d in data], dtype=np.float32),
            [d[4] for d in data],
            np.array([d[5] for d in data]),  # folder ids
        )

    for name, data in [('train',train_data),('val',val_data),('test',test_data)]:
        labs = np.array([d[1] for d in data])
        u, c = np.unique(labs, return_counts=True)
        fols = np.unique([d[5] for d in data])
        print(f"  {name:5s}: {len(data):6d} | folders={list(fols)} | "
              f"{ {CLASS_NAMES[int(uu)]:int(cc) for uu,cc in zip(u,c)} }")
    return to_arrays(train_data), to_arrays(val_data), to_arrays(test_data)

def load_ae_healthy():
    all_signals, all_rpms, all_fss, all_ffs = [], [], [], []
    for folder in AE_FOLDERS:
        result = load_mat(os.path.join(DATASET_PATH, folder, 'train.mat'))
        if result is None: continue
        measurements, fault_type, fault_freq = result
        if fault_type != 0: continue
        for sig, rpm, fs in measurements:
            if rpm<=0: rpm=1200.0
            if fs<=0:  fs=5120.0
            if len(sig)<WINDOW_SIZE: continue
            for start in range(0, len(sig)-WINDOW_SIZE+1, STEP):
                all_signals.append(sig[start:start+WINDOW_SIZE])
                all_rpms.append(rpm); all_fss.append(fs)
                all_ffs.append(fault_freq)
    return (np.array(all_signals,dtype=np.float32),
            np.array(all_rpms,dtype=np.float32),
            np.array(all_fss,dtype=np.float32),
            all_ffs)

# ── Physics ────────────────────────────────────────────────────────────────

def compute_all_order_physics(signals, rpms, fss, ffs):
    phys = []
    for i in range(len(signals)):
        p = compute_order_physics_features(
            signals[i], float(rpms[i]), float(fss[i]), ffs[i]
        )
        phys.append(p)
        if i % 10000 == 0: print(f"    {i}/{len(signals)}")
    return np.array(phys, dtype=np.float32)

# ── Dataset ────────────────────────────────────────────────────────────────

def augment_with_domain_sim(signal, rpm, fs, fault_freq):
    """
    Domain simulation augmentation.
    Simulate different machines by:
    - RPM jitter ±10%
    - Frequency warping ±10%
    """
    # RPM jitter ±10%
    rpm = rpm * np.random.uniform(0.90, 1.10)

    # Amplitude scaling ±20%
    signal = signal * np.random.uniform(0.80, 1.20)

    # Frequency warping ±10% via resampling
    warp = np.random.uniform(0.90, 1.10)
    if abs(warp - 1.0) > 0.01:
        n_orig = len(signal)
        n_new  = int(n_orig * warp)
        indices = np.linspace(0, n_orig-1, n_new)
        signal_warped = np.interp(indices, np.arange(n_orig), signal)
        # Crop or pad to original length
        if len(signal_warped) >= n_orig:
            signal = signal_warped[:n_orig].astype(np.float32)
        else:
            signal = np.pad(signal_warped, (0, n_orig-len(signal_warped))).astype(np.float32)

    return signal.astype(np.float32), float(rpm)

def inject_kurtosis_aug(signal, rpm, fs, fault_freq, fault_type):
    if fault_type == 0 or np.random.rand() > 0.4: return signal
    shaft_freq = rpm / 60.0
    fault_hz_map = {
        1: fault_freq['BPFIMultiple'] * shaft_freq,
        2: fault_freq['BPFMultiple']  * shaft_freq,
        3: fault_freq['BPFOMultiple'] * shaft_freq,
    }
    fault_hz = fault_hz_map.get(fault_type, 0)
    if fault_hz <= 0 or fault_hz >= fs/2: return signal
    period = int(fs / fault_hz)
    if period < 10: return signal
    impulse = np.zeros(len(signal), dtype=np.float32)
    for idx in range(0, len(signal), period):
        dlen = min(period//2, len(signal)-idx)
        decay = np.exp(-5 * np.arange(dlen) / dlen)
        impulse[idx:idx+dlen] += decay
    scale = np.sqrt(np.mean(signal**2)) * np.random.uniform(0.2, 0.6)
    return (signal + impulse * scale).astype(np.float32)

class FinalDataset(BearingDataset):
    def __init__(self, signals, labels, rpms, fss, ffs, folders,
                 phys_pre=None, augment=False):
        super().__init__(signals, labels, rpms, fss, ffs, augment=augment)
        self.folders  = folders
        self.phys_pre = phys_pre

    def __getitem__(self, idx):
        signal = self.signals[idx].copy()
        rpm    = float(self.rpm_arr[idx])
        fs     = float(self.fs_arr[idx])
        ff     = self.fault_freqs[idx]
        label  = int(self.labels[idx])
        folder = self.folders[idx]

        if self.augment:
            # Standard augmentation
            signal, rpm = augment_signal(signal, rpm, fs)
            # Domain simulation augmentation
            signal, rpm = augment_with_domain_sim(signal, rpm, fs, ff)
            # Kurtosis augmentation for fault classes
            signal = inject_kurtosis_aug(signal, rpm, fs, ff, label)

        sig_norm = normalize_signal(signal)
        x_time   = torch.from_numpy(sig_norm).float()
        x_freq   = torch.from_numpy(compute_order_stft(sig_norm, rpm, fs)).float()
        x_phys   = torch.from_numpy(self.phys_pre[idx]).float()
        phys_raw = torch.from_numpy(
            compute_order_physics_features(signal, rpm, fs, ff)
        ).float()
        folder_t = torch.tensor(int(folder), dtype=torch.long)
        return x_time, x_freq, x_phys, torch.tensor(label, dtype=torch.long), phys_raw, folder_t

# ── Model ──────────────────────────────────────────────────────────────────

class FinalModel(nn.Module):
    def __init__(self, num_classes=4, feature_dim=256):
        super().__init__()
        self.time_branch    = TimeDomainBranch(out_dim=feature_dim)
        self.freq_branch    = FrequencyBranch(out_dim=feature_dim)
        self.physics_branch = PhysicsBranch(input_dim=30, out_dim=feature_dim)
        self.fusion         = AttentionFusion(feature_dim, 3, num_classes)

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

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ── Evaluation ─────────────────────────────────────────────────────────────

def evaluate(model, loader, device, desc=""):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            x_time,x_freq,x_phys,y = batch[0],batch[1],batch[2],batch[3]
            logits,_ = model(x_time.to(device), x_freq.to(device), x_phys.to(device))
            probs = torch.softmax(logits, dim=-1)
            all_preds.extend(logits.argmax(-1).cpu().numpy())
            all_labels.extend(y.numpy())
            all_probs.extend(probs.cpu().numpy())
    p = np.array(all_preds); l = np.array(all_labels); pr = np.array(all_probs)
    acc   = accuracy_score(l, p)
    f1    = f1_score(l, p, average='macro', zero_division=0)
    prec  = precision_score(l, p, average='macro', zero_division=0)
    rec   = recall_score(l, p, average='macro', zero_division=0)
    mcc   = matthews_corrcoef(l, p)
    kappa = cohen_kappa_score(l, p)
    f1_per   = f1_score(l, p, average=None, zero_division=0)
    prec_per = precision_score(l, p, average=None, zero_division=0)
    rec_per  = recall_score(l, p, average=None, zero_division=0)
    cm = confusion_matrix(l, p)
    try:
        l_bin   = label_binarize(l, classes=[0,1,2,3])
        roc_auc = roc_auc_score(l_bin, pr, average='macro', multi_class='ovr')
    except: roc_auc = 0.0
    print(f"\n{'='*70}")
    print(f"RESULTS: {desc}")
    print(f"{'='*70}")
    print(f"  Acc:{acc:.4f} F1:{f1:.4f} Prec:{prec:.4f} Rec:{rec:.4f}")
    print(f"  AUC:{roc_auc:.4f} MCC:{mcc:.4f} Kappa:{kappa:.4f}")
    print(f"\n  {'Class':<20} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Sup':>6}")
    print(f"  {'-'*48}")
    for i, cls in enumerate(CLASS_NAMES):
        sup = int((l==i).sum())
        if sup > 0:
            print(f"  {cls:<20} {prec_per[i]:>6.4f} {rec_per[i]:>6.4f} "
                  f"{f1_per[i]:>6.4f} {sup:>6}")
    print(f"\n  Confusion Matrix:\n  {cm}")
    return {
        'accuracy': acc, 'f1_macro': f1, 'precision': prec, 'recall': rec,
        'roc_auc': roc_auc, 'mcc': mcc, 'kappa': kappa,
        'per_class': {CLASS_NAMES[i]: {
            'precision': float(prec_per[i]), 'recall': float(rec_per[i]),
            'f1': float(f1_per[i]), 'support': int((l==i).sum()),
        } for i in range(4)},
        'confusion_matrix': cm.tolist(),
    }

# ── Main ───────────────────────────────────────────────────────────────────

def train():
    print(f"Device: {DEVICE} | GPU: {torch.cuda.get_device_name(0)}")
    print(f"FINAL BEST — Order + Cross-folder Triplet + Domain Aug")
    print(f"Classifier: {CLASSIFIER_FOLDERS}, Hidden: {HIDDEN_FOLDERS}\n")

    print("Loading data...")
    (tr_sig,tr_lab,tr_rpm,tr_fs,tr_ff,tr_fol), \
    (va_sig,va_lab,va_rpm,va_fs,va_ff,va_fol), \
    (te_sig,te_lab,te_rpm,te_fs,te_ff,te_fol) = load_classifier_data()

    print("\nComputing order physics features...")
    print("  Train:")
    tr_phys_raw = compute_all_order_physics(tr_sig, tr_rpm, tr_fs, tr_ff)
    print("  Val:")
    va_phys_raw = compute_all_order_physics(va_sig, va_rpm, va_fs, va_ff)
    print("  Test:")
    te_phys_raw = compute_all_order_physics(te_sig, te_rpm, te_fs, te_ff)

    scaler = StandardScaler()
    tr_phys = scaler.fit_transform(tr_phys_raw).astype(np.float32)
    va_phys = scaler.transform(va_phys_raw).astype(np.float32)
    te_phys = scaler.transform(te_phys_raw).astype(np.float32)
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'physics_scaler_final_best.pkl'))

    def make_ds(sig, lab, rpm, fs, ff, fol, phys, aug):
        return FinalDataset(sig, lab, rpm, fs, ff, fol, phys_pre=phys, augment=aug)

    # Cross-folder balanced sampler for training
    tr_sampler = CrossFolderBalancedSampler(
        tr_lab, tr_fol, n_per_class=BATCH_SIZE//4, n_classes=4
    )
    tr_loader = DataLoader(
        make_ds(tr_sig,tr_lab,tr_rpm,tr_fs,tr_ff,tr_fol,tr_phys,True),
        batch_size=BATCH_SIZE, sampler=tr_sampler,
        num_workers=4, pin_memory=True
    )
    va_loader = DataLoader(
        make_ds(va_sig,va_lab,va_rpm,va_fs,va_ff,va_fol,va_phys,False),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )
    te_loader = DataLoader(
        make_ds(te_sig,te_lab,te_rpm,te_fs,te_ff,te_fol,te_phys,False),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )

    model      = FinalModel().to(DEVICE)
    triplet    = CrossFolderTripletLoss(margin=MARGIN)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    scaler_amp = torch.amp.GradScaler('cuda')
    print(f"\nModel parameters: {model.count_parameters():,}")
    print(f"Loss = CE + {LAMBDA_PHYS}×physics + {LAMBDA_TRIP}×cross_folder_triplet")

    best_f1, patience_cnt = 0.0, 0
    history = {'train_loss':[], 'val_f1':[], 'ce_loss':[], 'trip_loss':[]}

    print("\nTraining with cross-folder balanced batches...")
    for epoch in range(1, EPOCHS+1):
        model.train()
        epoch_loss = epoch_ce = epoch_trip = 0.0

        for batch in tr_loader:
            x_time,x_freq,x_phys,y,phys_raw,folders = (
                batch[0].to(DEVICE), batch[1].to(DEVICE), batch[2].to(DEVICE),
                batch[3].to(DEVICE), batch[4].to(DEVICE), batch[5].to(DEVICE)
            )
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                logits, _ = model(x_time, x_freq, x_phys)
                loss_ce, ce, _ = combined_loss(logits, y, phys_raw, LAMBDA_PHYS, LABEL_SMOOTH)

                # Cross-folder semi-hard triplet loss
                fused = model.get_fused(x_time, x_freq, x_phys)
                loss_trip = triplet(fused, y, folders)

                total_loss = loss_ce + LAMBDA_TRIP * loss_trip

            scaler_amp.scale(total_loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()

            epoch_loss += total_loss.item()
            epoch_ce   += ce
            epoch_trip += loss_trip.item()

        scheduler.step()

        model.eval()
        vp, vl = [], []
        with torch.no_grad():
            for batch in va_loader:
                x_time,x_freq,x_phys,y = batch[0],batch[1],batch[2],batch[3]
                logits,_ = model(x_time.to(DEVICE), x_freq.to(DEVICE), x_phys.to(DEVICE))
                vp.extend(logits.argmax(-1).cpu().numpy())
                vl.extend(y.numpy())
        val_f1  = f1_score(vl, vp, average='macro', zero_division=0)
        val_acc = accuracy_score(vl, vp)
        n = len(tr_loader)

        print(f"Epoch {epoch:3d} | Loss:{epoch_loss/n:.4f} CE:{epoch_ce/n:.4f} "
              f"T:{epoch_trip/n:.4f} | Acc:{val_acc:.4f} F1:{val_f1:.4f}")

        history['train_loss'].append(epoch_loss/n)
        history['val_f1'].append(val_f1)
        history['trip_loss'].append(epoch_trip/n)

        if val_f1 > best_f1:
            best_f1, patience_cnt = val_f1, 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_f1': val_f1, 'val_acc': val_acc,
                'classifier_folders': CLASSIFIER_FOLDERS,
                'hidden_folders': HIDDEN_FOLDERS,
                'method': 'order+cross_folder_triplet+domain_aug',
            }, os.path.join(MODELS_DIR, 'best_model_final_best.pth'))
            print(f"  ✓ Saved (F1={val_f1:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    ckpt = torch.load(os.path.join(MODELS_DIR, 'best_model_final_best.pth'))
    model.load_state_dict(ckpt['model_state_dict'])

    clf_results = evaluate(model, te_loader, DEVICE,
                           "FINAL BEST — INTERNAL TEST (folders 1-7)")

    # Train autoencoder
    print("\nTraining autoencoder...")
    ae_sigs, ae_rpms, ae_fss, ae_ffs = load_ae_healthy()
    model.eval()
    healthy_features = []
    for i in range(0, len(ae_sigs), 256):
        bs=ae_sigs[i:i+256]; br=ae_rpms[i:i+256]; bf=ae_fss[i:i+256]; bff=ae_ffs[i:i+256]
        x_t_list,x_f_list,x_p_list=[],[],[]
        for j in range(len(bs)):
            sig=bs[j]; rpm=float(br[j]); fs=float(bf[j]); ff=bff[j]
            phys=compute_order_physics_features(sig,rpm,fs,ff)
            ps=scaler.transform(phys.reshape(1,-1)).astype(np.float32)
            sn=normalize_signal(sig)
            ostft=compute_order_stft(sn,rpm,fs)
            x_t_list.append(sn); x_f_list.append(ostft); x_p_list.append(ps.flatten())
        x_t=torch.from_numpy(np.array(x_t_list)).float().to(DEVICE)
        x_f=torch.from_numpy(np.array(x_f_list)).float().to(DEVICE)
        x_p=torch.from_numpy(np.array(x_p_list)).float().to(DEVICE)
        with torch.no_grad():
            fused=model.get_fused(x_t,x_f,x_p)
        healthy_features.extend(fused.cpu().numpy())
        if i%5000==0: print(f"    {i}/{len(ae_sigs)}")
    healthy_features=np.array(healthy_features,dtype=np.float32)

    ae=BearingAutoencoder(input_dim=256,latent_dim=32).to(DEVICE)
    ae_opt=torch.optim.Adam(ae.parameters(),lr=1e-3,weight_decay=1e-5)
    ae_sch=torch.optim.lr_scheduler.CosineAnnealingLR(ae_opt,T_max=100,eta_min=1e-5)
    hf_t=torch.from_numpy(healthy_features).float()
    ae_ds=torch.utils.data.TensorDataset(hf_t)
    ae_ld=DataLoader(ae_ds,batch_size=256,shuffle=True)
    best_ae=float('inf')
    for ep in range(1,101):
        ae.train()
        el=0.0
        for (x,) in ae_ld:
            x=x.to(DEVICE); ae_opt.zero_grad()
            r,_=ae(x); loss=nn.MSELoss()(r,x)
            loss.backward(); ae_opt.step(); el+=loss.item()
        ae_sch.step(); avg=el/len(ae_ld)
        if ep%10==0: print(f"  AE {ep}/100 | Loss:{avg:.6f}")
        if avg<best_ae:
            best_ae=avg
            torch.save(ae.state_dict(),os.path.join(MODELS_DIR,'ae_final_best_tmp.pth'))
    ae.load_state_dict(torch.load(os.path.join(MODELS_DIR,'ae_final_best_tmp.pth')))
    ae.compute_threshold(healthy_features,n_sigma=5.0)
    torch.save({
        'model_state_dict':ae.state_dict(),
        'mean_error':ae.mean_error,'std_error':ae.std_error,'threshold':ae.threshold,
    },os.path.join(MODELS_DIR,'ae_final_best.pth'))

    results={'setup':{'method':'order+cross_folder_triplet+domain_aug+semi_hard_negatives',
                      'classifier_folders':CLASSIFIER_FOLDERS,'hidden_folders':HIDDEN_FOLDERS},
             'classifier':clf_results,'history':history}
    with open(os.path.join(RESULTS_DIR,'final_best_results.json'),'w') as f:
        json.dump(results,f,indent=2)

    print(f"\n{'='*70}")
    print(f"FINAL BEST SYSTEM READY")
    print(f"  Classifier:  best_model_final_best.pth")
    print(f"  Autoencoder: ae_final_best.pth")
    print(f"  Scaler:      physics_scaler_final_best.pkl")

if __name__ == '__main__':
    train()
