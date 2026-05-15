"""
ReconVLA baseline (plan_014).

Scalar Euclidean nonconformity score + argmin selection over K flow-matching samples.
No learned CRM, no per-DOF features — the simplest conformal selection baseline.
"""
import torch


def reconvla_select(k_samples, calibration_mean=None):
    """
    Select the best action from K samples via scalar Euclidean nonconformity.

    Args:
        k_samples: (K, H_EFF, ACTION_DIM) — K flow-matching action samples
        calibration_mean: (H_EFF, ACTION_DIM) — reference mean.
            If None, uses mean of k_samples (online estimation, consistent with ReconVLA).

    Returns:
        selected_action: (H_EFF, ACTION_DIM) — sample with minimum nonconformity score
    """
    K = k_samples.shape[0]
    if calibration_mean is None:
        calibration_mean = k_samples.mean(dim=0)

    flat_samples = k_samples.reshape(K, -1)
    flat_mean = calibration_mean.reshape(1, -1)

    scores = torch.norm(flat_samples - flat_mean, p=2, dim=1)
    best_idx = scores.argmin()
    return k_samples[best_idx]
