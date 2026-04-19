import numpy as np
import torch
from src.preprocessing.signal_processor import (
    normalize_signal, compute_stft, compute_physics_features
)

CLASS_NAMES = ['Healthy', 'Inner Race Fault', 'Ball Fault',
               'Outer Race Fault', 'Unknown Fault']

UNCERTAINTY_THRESHOLD    = 0.15
PHYSICS_ENERGY_THRESHOLD = 0.01
SIGNAL_ENERGY_THRESHOLD  = 1e-4
AE_HEALTHY_THRESHOLD     = 3.0   # below this = healthy
AE_UNKNOWN_THRESHOLD     = 10.0  # above this + uncertain = unknown fault

FAULT_ENERGY_INDICES = {
    1: list(range(12,16)) + [18,19],  # inner race BPFI
    2: list(range(4,8)),               # ball BSF
    3: list(range(8,12)) + [16,17],   # outer race BPFO
}

def check_signal_quality(raw_signal):
    rms  = float(np.sqrt(np.mean(raw_signal**2)))
    std  = float(raw_signal.std())
    kurt = float(np.mean(((raw_signal - raw_signal.mean())/(std+1e-8))**4))
    return rms, kurt, rms < SIGNAL_ENERGY_THRESHOLD

def check_physics_consistency(pred_class, phys_raw, signal_rms):
    if pred_class == 0:
        return True, 0.0
    indices = FAULT_ENERGY_INDICES.get(pred_class, [])
    if not indices:
        return True, 0.0
    fault_energy = float(phys_raw[indices].mean())
    relative     = fault_energy / (signal_rms + 1e-8)
    return relative > PHYSICS_ENERGY_THRESHOLD, fault_energy

def get_fused_features(model, x_time, x_freq, x_phys, device):
    """Extract 256-dim fused representation from 3-branch model"""
    model.eval()
    with torch.no_grad():
        f_time  = model.time_branch(x_time)
        f_freq  = model.freq_branch(x_freq)
        f_phys  = model.physics_branch(x_phys)
        concat  = torch.cat([f_time, f_freq, f_phys], dim=-1)
        weights = model.fusion.attention(concat)
        stacked = torch.stack([f_time, f_freq, f_phys], dim=1)
        fused   = (stacked * weights.unsqueeze(-1)).sum(dim=1)
    return fused

def full_inference(model, scaler, autoencoder,
                   raw_signal, rpm, fs, ff,
                   device, n_mc=50, label="Signal"):
    """
    Two-stage inference pipeline:

    STAGE 1 — Autoencoder (is this healthy or anomaly?)
      AE error < 3σ  → HEALTHY confirmed
      AE error > 3σ  → anomaly, go to Stage 2

    STAGE 2 — Classifier (which fault is it?)
      MC Dropout → prediction + uncertainty
      uncertainty > 0.15 → UNCERTAIN
      AE error > 10σ AND uncertain → UNKNOWN FAULT
      physics mismatch → PHYSICS MISMATCH
      all pass → CONFIDENT diagnosis
    """

    # ── Pre-checks ────────────────────────────────────────────────────────
    rms, kurt, low_energy = check_signal_quality(raw_signal)
    if low_energy:
        return _build_result(label, -1, 'Unknown', 0.0, 0.0,
                             {}, rms, kurt, True, True, 0.0,
                             None, "DEAD SIGNAL — sensor failure or disconnected",
                             False, 'dead_signal', raw_signal)

    # Physics features on RAW signal before normalization
    phys_raw   = compute_physics_features(raw_signal, rpm, fs, ff)
    phys_scaled = scaler.transform(phys_raw.reshape(1,-1)).astype(np.float32)

    # Normalize signal for model input
    sig_norm = normalize_signal(raw_signal)
    x_time   = torch.from_numpy(sig_norm).unsqueeze(0).to(device)
    x_freq   = torch.from_numpy(compute_stft(sig_norm, fs)).unsqueeze(0).to(device)
    x_phys   = torch.from_numpy(phys_scaled).to(device)

    # ── STAGE 1 — Autoencoder check ───────────────────────────────────────
    fused = get_fused_features(model, x_time, x_freq, x_phys, device)

    ae_norm_error = None
    if autoencoder is not None:
        autoencoder.eval()
        with torch.no_grad():
            ae_norm_error = float(autoencoder.normalized_error(fused).item())

        if ae_norm_error < AE_HEALTHY_THRESHOLD:
            # Autoencoder says this looks healthy
            # Still run quick classifier check to confirm
            model.eval()
            with torch.no_grad():
                logits, weights = model(x_time, x_freq, x_phys)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            pred_class = int(probs.argmax())
            confidence = float(probs[pred_class])
            branch_w   = weights.cpu().numpy()[0]

            return _build_result(
                label, 0, 'Healthy', confidence, 0.0,
                {'time': float(branch_w[0]),
                 'frequency': float(branch_w[1]),
                 'physics': float(branch_w[2])},
                rms, kurt, low_energy, False, 0.0,
                ae_norm_error,
                f"HEALTHY — AE reconstruction error {ae_norm_error:.2f}σ "
                f"(below {AE_HEALTHY_THRESHOLD}σ threshold). "
                f"No anomaly detected.",
                True, 'healthy_confirmed', raw_signal
            )

    # ── STAGE 2 — Classifier (anomaly detected, identify fault type) ──────
    def enable_dropout(m):
        if isinstance(m, torch.nn.Dropout): m.train()

    model.eval()
    model.apply(enable_dropout)

    mc_preds, mc_weights = [], []
    xb_t = x_time.repeat(32, 1)
    xb_f = x_freq.repeat(32, 1, 1)
    xb_p = x_phys.repeat(32, 1)

    with torch.no_grad():
        for _ in range(n_mc):
            logits, weights = model(xb_t, xb_f, xb_p)
            mc_preds.append(torch.softmax(logits, dim=-1).cpu().numpy()[0])
            mc_weights.append(weights.cpu().numpy()[0])

    mc_preds     = np.array(mc_preds)
    mean_pred    = mc_preds.mean(axis=0)
    pred_class   = int(mean_pred.argmax())
    confidence   = float(mean_pred[pred_class])
    uncertainty  = float(mc_preds.std(axis=0)[pred_class])
    mean_weights = np.array(mc_weights).mean(axis=0)

    branch_weights = {
        'time':      float(mean_weights[0]),
        'frequency': float(mean_weights[1]),
        'physics':   float(mean_weights[2]),
    }

    # Physics consistency check
    physics_consistent, fault_energy = check_physics_consistency(
        pred_class, phys_raw, rms
    )

    # Decision logic
    # Check unknown fault first — very high AE error + uncertain
    if (ae_norm_error is not None and
            ae_norm_error > AE_UNKNOWN_THRESHOLD and
            uncertainty > UNCERTAINTY_THRESHOLD):
        decision = (
            f"UNKNOWN FAULT — AE error {ae_norm_error:.1f}σ far above healthy "
            f"baseline AND model uncertain (±{uncertainty:.3f}). "
            f"Pattern never seen in training. "
            f"Escalate to senior engineer immediately."
        )
        return _build_result(label, 4, 'Unknown Fault', confidence, uncertainty,
                             branch_weights, rms, kurt, low_energy,
                             False, fault_energy, ae_norm_error,
                             decision, False, 'unknown_fault', raw_signal)

    # Unknown fault — very high AE error + physics mismatch
    if (ae_norm_error is not None and
            ae_norm_error > AE_UNKNOWN_THRESHOLD and
            not physics_consistent):
        decision = (
            f"UNKNOWN FAULT — AE error {ae_norm_error:.1f}σ and physics "
            f"inconsistent. This fault type was not seen during training. "
            f"Please label and retrain."
        )
        return _build_result(label, 4, 'Unknown Fault', confidence, uncertainty,
                             branch_weights, rms, kurt, low_energy,
                             False, fault_energy, ae_norm_error,
                             decision, False, 'unknown_fault', raw_signal)

    # Uncertain prediction
    if uncertainty > UNCERTAINTY_THRESHOLD:
        decision = (
            f"UNCERTAIN (±{uncertainty:.3f}) — anomaly detected "
            f"(AE={ae_norm_error:.1f}σ) but classifier is internally "
            f"inconsistent. Recommend physical inspection."
        )
        return _build_result(label, pred_class, CLASS_NAMES[pred_class],
                             confidence, uncertainty, branch_weights,
                             rms, kurt, low_energy, physics_consistent,
                             fault_energy, ae_norm_error,
                             decision, False, 'uncertain', raw_signal)

    # Physics mismatch
    if pred_class != 0 and not physics_consistent:
        rel = fault_energy / (rms + 1e-8)
        decision = (
            f"PHYSICS MISMATCH — classifier predicts "
            f"{CLASS_NAMES[pred_class]} but fault frequency energy "
            f"is too low (relative={rel:.4f}). "
            f"AE detected anomaly ({ae_norm_error:.1f}σ) but spectral "
            f"evidence is insufficient. Recommend inspection."
        )
        return _build_result(label, pred_class, CLASS_NAMES[pred_class],
                             confidence, uncertainty, branch_weights,
                             rms, kurt, low_energy, physics_consistent,
                             fault_energy, ae_norm_error,
                             decision, False, 'physics_mismatch', raw_signal)

    # All checks passed — confident known fault diagnosis
    ae_str = f", AE={ae_norm_error:.1f}σ" if ae_norm_error is not None else ""
    decision = (
        f"CONFIDENT — {CLASS_NAMES[pred_class]} ({confidence:.1%}) "
        f"[uncertainty=±{uncertainty:.3f}{ae_str}, "
        f"physics={'✓' if physics_consistent else '⚠'}]"
    )
    return _build_result(label, pred_class, CLASS_NAMES[pred_class],
                         confidence, uncertainty, branch_weights,
                         rms, kurt, low_energy, physics_consistent,
                         fault_energy, ae_norm_error,
                         decision, True, 'confident', raw_signal)


def _build_result(label, pred_class, pred_name, confidence, uncertainty,
                  branch_weights, rms, kurt, low_energy, physics_consistent,
                  fault_energy, ae_norm_error, decision, reliable,
                  decision_type, raw_signal):
    probs = {}
    return {
        'label':               label,
        'pred_class':          pred_class,
        'pred_name':           pred_name,
        'confidence':          confidence,
        'uncertainty':         uncertainty,
        'branch_weights':      branch_weights,
        'signal_rms':          rms,
        'signal_kurtosis':     kurt,
        'low_energy':          low_energy,
        'physics_consistent':  physics_consistent,
        'fault_energy':        fault_energy,
        'ae_normalized_error': ae_norm_error,
        'decision':            decision,
        'decision_type':       decision_type,
        'reliable':            reliable,
    }


def print_result(r):
    stage = "STAGE 1 (AE)" if r['decision_type'] in ['healthy_confirmed','dead_signal'] \
            else "STAGE 2 (Classifier)"
    print(f"\n  [{r['label']}] — {stage}")
    print(f"    Signal RMS:   {r['signal_rms']:.6f} | Kurtosis: {r['signal_kurtosis']:.2f}")
    if r['ae_normalized_error'] is not None:
        ae_status = '✓ HEALTHY' if r['ae_normalized_error'] < AE_HEALTHY_THRESHOLD \
                    else '⚠ ANOMALY'
        print(f"    AE error:     {r['ae_normalized_error']:.2f}σ — {ae_status}")
    if r['decision_type'] not in ['healthy_confirmed','dead_signal']:
        print(f"    Classifier:   {r['pred_name']} ({r['confidence']:.1%})")
        print(f"    Uncertainty:  ±{r['uncertainty']:.4f} "
              f"{'⚠ HIGH' if r['uncertainty'] > UNCERTAINTY_THRESHOLD else '✓ OK'}")
        print(f"    Physics:      {'✓' if r['physics_consistent'] else '⚠ mismatch'} "
              f"(energy={r['fault_energy']:.6f})")
        if r['branch_weights']:
            bw = r['branch_weights']
            print(f"    Branches:     time={bw.get('time',0):.3f} "
                  f"freq={bw.get('frequency',0):.3f} "
                  f"phys={bw.get('physics',0):.3f}")
    print(f"    FINAL:        {r['decision']}")


def get_maintenance_recommendation(decision_type, pred_class):
    """Plain English maintenance recommendation"""
    recs = {
        'healthy_confirmed': "Machine is operating normally. Continue scheduled monitoring.",
        'dead_signal':       "Check sensor connection immediately. No diagnosis possible.",
        'unknown_fault':     "Unknown fault pattern detected. Stop machine and inspect physically. Label this fault for model retraining.",
        'uncertain':         "Anomaly detected but type unclear. Schedule inspection within 24 hours.",
        'physics_mismatch':  "Anomaly detected. Physical inspection recommended within 48 hours.",
        'confident': {
            1: "Inner Race Fault confirmed. Schedule bearing replacement within 48-72 hours.",
            2: "Ball Fault confirmed. Schedule bearing replacement within 48-72 hours.",
            3: "Outer Race Fault confirmed. Schedule bearing replacement within 24-48 hours.",
            0: "Machine is healthy. Continue normal operation.",
        }
    }
    if decision_type == 'confident':
        return recs['confident'].get(pred_class, "Schedule inspection.")
    return recs.get(decision_type, "Schedule inspection.")
