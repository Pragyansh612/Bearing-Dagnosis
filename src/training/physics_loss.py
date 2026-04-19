import torch
import torch.nn.functional as F

# Maps class index to which physics feature indices should be elevated
# Physics features layout (from signal_processor.py):
# indices 0-3:  FTF harmonics
# indices 4-7:  BPF harmonics
# indices 8-11: BPFO harmonics  ← outer race fault (class 3)
# indices 12-15: BPFI harmonics ← inner race fault (class 1)
# indices 16-17: BPFO sidebands ← outer race fault (class 3)
# indices 18-19: BPFI sidebands ← inner race fault (class 1)
# indices 4-7:  BPF harmonics   ← ball fault (class 2)

FAULT_PHYSICS_INDICES = {
    1: [12, 13, 14, 15, 18, 19],  # inner race: BPFI harmonics + sidebands
    2: [4,  5,  6,  7],           # ball: BPF harmonics
    3: [8,  9,  10, 11, 16, 17],  # outer race: BPFO harmonics + sidebands
}

def physics_consistency_penalty(probs, physics_features, labels=None, lambda_val=0.3):
    """
    Penalize predictions that are inconsistent with physical evidence.
    If model predicts fault class X but the corresponding fault frequency
    energy is low, apply a penalty.
    """
    penalty = torch.tensor(0.0, device=probs.device, requires_grad=True)
    count = 0

    for fault_class, freq_indices in FAULT_PHYSICS_INDICES.items():
        # Probability of predicting this fault class
        pred_prob = probs[:, fault_class]  # (B,)

        # Spectral energy at this fault's characteristic frequencies
        indices = torch.tensor(freq_indices, device=physics_features.device)
        fault_energy = physics_features[:, indices].mean(dim=-1)  # (B,)

        # Normalize energy using sigmoid
        normalized_energy = torch.sigmoid(fault_energy * 10.0)

        # Penalty: high prediction confidence + low physical evidence
        inconsistency = pred_prob * (1.0 - normalized_energy)
        penalty = penalty + inconsistency.mean()
        count += 1

    return penalty / max(count, 1)

def combined_loss(logits, targets, physics_features, lambda_physics=0.3,
                  label_smoothing=0.1):
    # Cross entropy with label smoothing
    ce_loss = F.cross_entropy(logits, targets, label_smoothing=label_smoothing)

    # Physics penalty
    probs = F.softmax(logits, dim=-1)
    phys_penalty = physics_consistency_penalty(probs, physics_features)

    total = ce_loss + lambda_physics * phys_penalty
    return total, ce_loss.item(), phys_penalty.item()
