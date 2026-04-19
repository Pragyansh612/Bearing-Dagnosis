import os, sys, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, confusion_matrix, roc_auc_score,
                              average_precision_score)
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
BATCH_SIZE   = 128
EPOCHS       = 100
LR           = 1e-3
WEIGHT_DECAY = 1e-4
LAMBDA_PHYS  = 0.5
LAMBDA_TRIP  = 0.1   # triplet loss weight
LAMBDA_CORAL = 0.0   # CORAL loss weight during training
LABEL_SMOOTH = 0.1
PATIENCE     = 15
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_DIR  = os.path.expanduser('~/bearing_diagnosis/results')
MODELS_DIR   = os.path.expanduser('~/bearing_diagnosis/models')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

CLASS_SPECIFIC_WEIGHTS = {0: 1.0, 1: 1.5, 2: 1.2, 3: 1.5}

# ── Triplet Loss ───────────────────────────────────────────────────────────

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, features, labels):
        """
        Semi-hard triplet mining.
        Forces: same fault type → close in feature space
                different fault → far apart
        """
        device = features.device
        labels_np = labels.cpu().numpy()
        n = len(labels_np)

        anchors, positives, negatives = [], [], []
        for i in range(n):
            pos_idx = [j for j in range(n)
                       if labels_np[j] == labels_np[i] and j != i]
            neg_idx = [j for j in range(n)
                       if labels_np[j] != labels_np[i]]
            if not pos_idx or not neg_idx:
                continue
            p = pos_idx[np.random.randint(len(pos_idx))]
            q = neg_idx[np.random.randint(len(neg_idx))]
            anchors.append(i); positives.append(p); negatives.append(q)

        if not anchors:
            return torch.tensor(0.0, device=device)

        a = features[torch.tensor(anchors, device=device)]
        p = features[torch.tensor(positives, device=device)]
        n_ = features[torch.tensor(negatives, device=device)]

        pos_dist = F.pairwise_distance(a, p)
        neg_dist = F.pairwise_distance(a, n_)
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()

# ── CORAL Loss (during training) ───────────────────────────────────────────

def coral_loss(source_features, target_features):
    """
    CORAL loss computed during training.
    Minimizes difference between source and target covariance matrices.
    When applied during training, model learns to produce
    distribution-aligned features automatically.
    """
    ns = source_features.shape[0]
    nt = target_features.shape[0]
    d  = source_features.shape[1]

    # Source covariance
    s_mean = source_features.mean(dim=0, keepdim=True)
    s_cov  = (source_features - s_mean).T @ (source_features - s_mean) / (ns - 1)

    # Target covariance
    t_mean = target_features.mean(dim=0, keepdim=True)
    t_cov  = (target_features - t_mean).T @ (target_features - t_mean) / (nt - 1)

    # Frobenius norm of covariance difference
    loss = torch.norm(s_cov - t_cov, p='fro') ** 2
    return loss / (4 * d * d)

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

def to_windows(measurements, fault_type, fault_freq):
    windows = []
    for sig, rpm, fs in measurements:
        if rpm<=0: rpm=1200.0
        if fs<=0:  fs=5120.0
        if len(sig)<WINDOW_SIZE: continue
        for start in range(0, len(sig)-WINDOW_SIZE+1, STEP):
            w = sig[start:start+WINDOW_SIZE]
            windows.append((w, fault_type, rpm, fs, fault_freq))
    return windows

# ── Data loading ───────────────────────────────────────────────────────────

def load_classifier_data():
    train_data, val_data, test_data = [], [], []
    for folder in CLASSIFIER_FOLDERS:
        folder_path = os.path.join(DATASET_PATH, folder)
        result = load_mat(os.path.join(folder_path, 'train.mat'))
        if result:
            measurements, fault_type, fault_freq = result
            n = len(measurements)
            n_tr = int(n*0.85); n_va = int(n*0.075)
            train_data.extend(to_windows(measurements[:n_tr], fault_type, fault_freq))
            val_data.extend(to_windows(measurements[n_tr:n_tr+n_va], fault_type, fault_freq))
            test_data.extend(to_windows(measurements[n_tr+n_va:], fault_type, fault_freq))
        result = load_mat(os.path.join(folder_path, 'test.mat'))
        if result:
            measurements, fault_type, fault_freq = result
            n = len(measurements)
            n_tr = int(n*0.70); n_va = int(n*0.15)
            train_data.extend(to_windows(measurements[:n_tr], fault_type, fault_freq))
            val_data.extend(to_windows(measurements[n_tr:n_tr+n_va], fault_type, fault_freq))
            test_data.extend(to_windows(measurements[n_tr+n_va:], fault_type, fault_freq))

    def to_arrays(data):
        return (
            np.array([d[0] for d in data], dtype=np.float32),
            np.array([d[1] for d in data], dtype=np.int64),
            np.array([d[2] for d in data], dtype=np.float32),
            np.array([d[3] for d in data], dtype=np.float32),
            [d[4] for d in data],
        )

    for name, data in [('train',train_data),('val',val_data),('test',test_data)]:
        labs = np.array([d[1] for d in data])
        u, c = np.unique(labs, return_counts=True)
        print(f"  {name:5s}: {len(data):6d} | "
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
                w = sig[start:start+WINDOW_SIZE]
                all_signals.append(w)
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

class OrderDataset(BearingDataset):
    """Dataset using order-domain features — machine invariant"""
    def __init__(self, *args, phys_pre=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.phys_pre = phys_pre

    def __getitem__(self, idx):
        signal = self.signals[idx].copy()
        rpm    = float(self.rpm_arr[idx])
        fs     = float(self.fs_arr[idx])
        ff     = self.fault_freqs[idx]
        label  = int(self.labels[idx])

        if self.augment:
            signal, rpm = augment_signal(signal, rpm, fs)
            signal = inject_kurtosis_aug(signal, rpm, fs, ff, label)

        # ORDER-DOMAIN STFT — machine invariant
        sig_norm  = normalize_signal(signal)
        x_time    = torch.from_numpy(sig_norm).float()
        x_freq    = torch.from_numpy(
            compute_order_stft(sig_norm, rpm, fs)
        ).float()
        x_phys    = torch.from_numpy(self.phys_pre[idx]).float()
        phys_raw  = torch.from_numpy(
            compute_order_physics_features(signal, rpm, fs, ff)
        ).float()
        return x_time, x_freq, x_phys, torch.tensor(label, dtype=torch.long), phys_raw

def get_sampler(labels):
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    freq_w = {int(u): total/(len(unique)*c) for u,c in zip(unique,counts)}
    combined = {c: freq_w.get(c,1.0)*CLASS_SPECIFIC_WEIGHTS.get(c,1.0) for c in range(4)}
    sw = [combined[int(l)] for l in labels]
    return WeightedRandomSampler(sw, len(sw))

# ── Model ──────────────────────────────────────────────────────────────────

class OrderModel(nn.Module):
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
        for x_time,x_freq,x_phys,y,_ in loader:
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
    print(f"ORDER-DOMAIN + TRIPLET LOSS + CORAL TRAINING")
    print(f"Classifier: {CLASSIFIER_FOLDERS}")
    print(f"Hidden:     {HIDDEN_FOLDERS}\n")

    print("Loading data...")
    (tr_sig,tr_lab,tr_rpm,tr_fs,tr_ff), \
    (va_sig,va_lab,va_rpm,va_fs,va_ff), \
    (te_sig,te_lab,te_rpm,te_fs,te_ff) = load_classifier_data()

    print("\nComputing order-domain physics features...")
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
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'physics_scaler_order.pkl'))

    def make_ds(sig, lab, rpm, fs, ff, phys, aug):
        return OrderDataset(sig, lab, rpm, fs, ff, augment=aug, phys_pre=phys)

    tr_loader = DataLoader(make_ds(tr_sig,tr_lab,tr_rpm,tr_fs,tr_ff,tr_phys,True),
                           batch_size=BATCH_SIZE, sampler=get_sampler(tr_lab),
                           num_workers=4, pin_memory=True)
    va_loader = DataLoader(make_ds(va_sig,va_lab,va_rpm,va_fs,va_ff,va_phys,False),
                           batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=4, pin_memory=True)
    te_loader = DataLoader(make_ds(te_sig,te_lab,te_rpm,te_fs,te_ff,te_phys,False),
                           batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=4, pin_memory=True)

    model      = OrderModel().to(DEVICE)
    triplet    = TripletLoss(margin=1.0)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    scaler_amp = torch.amp.GradScaler('cuda')
    print(f"\nModel parameters: {model.count_parameters():,}")
    print(f"Loss = CE + {LAMBDA_PHYS}×physics + {LAMBDA_TRIP}×triplet + {LAMBDA_CORAL}×CORAL")

    best_f1, patience_cnt = 0.0, 0
    history = {'train_loss':[], 'val_f1':[], 'ce_loss':[],
               'phys_loss':[], 'trip_loss':[], 'coral_loss':[]}

    print("\nTraining...")
    for epoch in range(1, EPOCHS+1):
        model.train()
        epoch_loss = 0.0
        epoch_ce = epoch_phys = epoch_trip = epoch_coral = 0.0

        for x_time,x_freq,x_phys,y,phys_raw in tr_loader:
            x_time   = x_time.to(DEVICE)
            x_freq   = x_freq.to(DEVICE)
            x_phys   = x_phys.to(DEVICE)
            y        = y.to(DEVICE)
            phys_raw = phys_raw.to(DEVICE)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                logits, _ = model(x_time, x_freq, x_phys)

                # CE + physics loss
                loss_ce, ce, phys = combined_loss(
                    logits, y, phys_raw, LAMBDA_PHYS, LABEL_SMOOTH
                )

                # Triplet loss on fused features
                fused = model.get_fused(x_time, x_freq, x_phys)
                loss_trip = triplet(fused, y)

                # CORAL loss — align fault class features
                # Use healthy vs fault as source/target split
                healthy_mask = (y == 0)
                fault_mask   = (y != 0)
                if healthy_mask.sum() > 1 and fault_mask.sum() > 1:
                    loss_coral = coral_loss(
                        fused[healthy_mask], fused[fault_mask]
                    )
                else:
                    loss_coral = torch.tensor(0.0, device=DEVICE)

                total_loss = (loss_ce +
                              LAMBDA_TRIP  * loss_trip +
                              LAMBDA_CORAL * loss_coral)

            scaler_amp.scale(total_loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()

            epoch_loss  += total_loss.item()
            epoch_ce    += ce
            epoch_phys  += phys
            epoch_trip  += loss_trip.item()
            epoch_coral += loss_coral.item()

        scheduler.step()

        model.eval()
        vp, vl = [], []
        with torch.no_grad():
            for x_time,x_freq,x_phys,y,_ in va_loader:
                logits,_ = model(x_time.to(DEVICE), x_freq.to(DEVICE), x_phys.to(DEVICE))
                vp.extend(logits.argmax(-1).cpu().numpy())
                vl.extend(y.numpy())
        val_f1  = f1_score(vl, vp, average='macro', zero_division=0)
        val_acc = accuracy_score(vl, vp)
        n = len(tr_loader)

        print(f"Epoch {epoch:3d} | "
              f"Loss:{epoch_loss/n:.4f} CE:{epoch_ce/n:.4f} "
              f"T:{epoch_trip/n:.4f} C:{epoch_coral/n:.4f} | "
              f"Acc:{val_acc:.4f} F1:{val_f1:.4f}")

        history['train_loss'].append(epoch_loss/n)
        history['val_f1'].append(val_f1)
        history['ce_loss'].append(epoch_ce/n)
        history['trip_loss'].append(epoch_trip/n)
        history['coral_loss'].append(epoch_coral/n)

        if val_f1 > best_f1:
            best_f1, patience_cnt = val_f1, 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_f1': val_f1, 'val_acc': val_acc,
                'classifier_folders': CLASSIFIER_FOLDERS,
                'hidden_folders': HIDDEN_FOLDERS,
                'features': 'order_domain',
                'losses': 'CE+physics+triplet+CORAL',
            }, os.path.join(MODELS_DIR, 'best_model_order.pth'))
            print(f"  ✓ Saved (F1={val_f1:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    ckpt = torch.load(os.path.join(MODELS_DIR, 'best_model_order.pth'))
    model.load_state_dict(ckpt['model_state_dict'])

    clf_results = evaluate(model, te_loader, DEVICE,
                           "ORDER MODEL — INTERNAL TEST (folders 1-7)")

    # ── Train autoencoder on order features ───────────────────────────────
    print("\n" + "="*60)
    print("AUTOENCODER on order features (folders 1-7 healthy)")
    print("="*60)

    ae_sigs, ae_rpms, ae_fss, ae_ffs = load_ae_healthy()
    print(f"  {len(ae_sigs)} healthy windows")

    model.eval()
    healthy_features = []
    for i in range(0, len(ae_sigs), 256):
        bs = ae_sigs[i:i+256]
        br = ae_rpms[i:i+256]
        bf = ae_fss[i:i+256]
        bff = ae_ffs[i:i+256]
        x_t_list, x_f_list, x_p_list = [], [], []
        for j in range(len(bs)):
            sig    = bs[j]; rpm = float(br[j]); fs = float(bf[j]); ff = bff[j]
            phys   = compute_order_physics_features(sig, rpm, fs, ff)
            phys_s = scaler.transform(phys.reshape(1,-1)).astype(np.float32)
            sn     = normalize_signal(sig)
            ostft  = compute_order_stft(sn, rpm, fs)
            x_t_list.append(sn)
            x_f_list.append(ostft)
            x_p_list.append(phys_s.flatten())
        x_t = torch.from_numpy(np.array(x_t_list)).float().to(DEVICE)
        x_f = torch.from_numpy(np.array(x_f_list)).float().to(DEVICE)
        x_p = torch.from_numpy(np.array(x_p_list)).float().to(DEVICE)
        with torch.no_grad():
            fused = model.get_fused(x_t, x_f, x_p)
        healthy_features.extend(fused.cpu().numpy())
        if i % 5000 == 0: print(f"    {i}/{len(ae_sigs)}")
    healthy_features = np.array(healthy_features, dtype=np.float32)
    print(f"  Features: {healthy_features.shape}")

    ae = BearingAutoencoder(input_dim=256, latent_dim=32).to(DEVICE)
    ae_optim = torch.optim.Adam(ae.parameters(), lr=1e-3, weight_decay=1e-5)
    ae_sched = torch.optim.lr_scheduler.CosineAnnealingLR(ae_optim, T_max=100, eta_min=1e-5)
    hf_t = torch.from_numpy(healthy_features).float()
    ae_ds = torch.utils.data.TensorDataset(hf_t)
    ae_ld = DataLoader(ae_ds, batch_size=256, shuffle=True)

    best_ae = float('inf')
    for ep in range(1, 101):
        ae.train()
        el = 0.0
        for (x,) in ae_ld:
            x = x.to(DEVICE)
            ae_optim.zero_grad()
            r, _ = ae(x)
            loss = nn.MSELoss()(r, x)
            loss.backward(); ae_optim.step()
            el += loss.item()
        ae_sched.step()
        avg = el/len(ae_ld)
        if ep % 10 == 0: print(f"  AE {ep}/100 | Loss:{avg:.6f}")
        if avg < best_ae:
            best_ae = avg
            torch.save(ae.state_dict(), os.path.join(MODELS_DIR, 'ae_order_best.pth'))

    ae.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'ae_order_best.pth')))
    ae.compute_threshold(healthy_features, n_sigma=5.0)
    torch.save({
        'model_state_dict': ae.state_dict(),
        'mean_error': ae.mean_error,
        'std_error':  ae.std_error,
        'threshold':  ae.threshold,
    }, os.path.join(MODELS_DIR, 'autoencoder_order.pth'))

    # Save results
    results = {
        'setup': {
            'features': 'order_domain',
            'losses': 'CE + physics + triplet + CORAL',
            'classifier_folders': CLASSIFIER_FOLDERS,
            'hidden_folders': HIDDEN_FOLDERS,
        },
        'classifier': clf_results,
        'ae_threshold': ae.threshold,
        'history': history,
    }
    with open(os.path.join(RESULTS_DIR, 'order_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"ORDER MODEL READY")
    print(f"  Classifier:  best_model_order.pth")
    print(f"  Autoencoder: autoencoder_order.pth")
    print(f"  Scaler:      physics_scaler_order.pkl")

if __name__ == '__main__':
    train()
