"""
Conformalized Gaussian Scoring (CGS) module.

Input: K flow-matching samples, shape (K, H_eff, D) where D=CORRECTION_DIM=6, H_eff=5.
       Accepts (K, chunk_size, D') and auto-truncates to (K, H_eff, action_dim).

Output:
  - uncertainty_features: (action_dim*H_eff + H_eff + H_eff) tensor
    [per_dof_std(30), conformal_radius(5), log_volume(5)]
  - mean_action: (H_eff*action_dim) tensor (K samples mean, flattened)
"""
import torch
import numpy as np
from sklearn.covariance import LedoitWolf

try:
    from utils import CORRECTION_DIM
except ImportError:
    from .utils import CORRECTION_DIM


class ConformilizedGaussianScoring:
    """Compute structured uncertainty features from K flow-matching samples (pose only)."""

    def __init__(self, K: int = 10, horizon: int = 5, action_dim: int = CORRECTION_DIM,
                 alpha: float = 0.5, chunk_size: int = 50):  # BugFix W3: 0.1→0.5, K=10时0.1导致quantile=max
        self.K = K
        self.horizon = horizon       # H_eff = 5
        self.action_dim = action_dim  # 6D pose (no gripper)
        self.alpha = alpha
        self.chunk_size = chunk_size

    def compute_features(self, samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            samples: (K, T, D') where T >= H_eff and D' >= action_dim.
                     Auto-truncates to (K, H_eff, action_dim).
        Returns:
            uncertainty_features: (action_dim*H_eff + H_eff + H_eff,)
            mean_action: (H_eff * action_dim,) — flattened mean over K samples
        """
        if samples.shape[1] > self.horizon:
            samples = samples[:, :self.horizon, :]
        if samples.shape[2] > self.action_dim:
            samples = samples[:, :, :self.action_dim]

        assert samples.shape == (self.K, self.horizon, self.action_dim), \
            f"Expected ({self.K}, {self.horizon}, {self.action_dim}), got {samples.shape}"

        mean_action = samples.mean(dim=0)  # (H, D)

        per_dof_stds = []
        conformal_radii = []
        log_volumes = []

        for h in range(self.horizon):
            step_samples = samples[:, h, :]  # (K, D)
            cov_matrix = self._ledoit_wolf_cov(step_samples)  # (D, D)

            stds = torch.sqrt(torch.diag(cov_matrix).clamp(min=1e-10))
            per_dof_stds.append(stds)

            q_h = self._conformal_radius(step_samples, mean_action[h], cov_matrix)
            conformal_radii.append(q_h)

            sign, logabsdet = torch.linalg.slogdet(cov_matrix)
            log_vol = 0.5 * logabsdet
            log_volumes.append(log_vol)

        # 6*5=30 + 5 + 5 = 40D
        uncertainty_features = torch.cat([
            torch.cat(per_dof_stds),       # (30,)
            torch.stack(conformal_radii),   # (5,)
            torch.stack(log_volumes),       # (5,)
        ])

        return uncertainty_features, mean_action.flatten()

    def compute_features_batch(self, samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch version: process B sets of K samples."""
        B = samples.shape[0]
        all_uf = []
        all_ma = []
        for b in range(B):
            uf, ma = self.compute_features(samples[b])
            all_uf.append(uf)
            all_ma.append(ma)
        return torch.stack(all_uf), torch.stack(all_ma)

    def _ledoit_wolf_cov(self, samples: torch.Tensor) -> torch.Tensor:
        samples_np = samples.detach().cpu().numpy()
        lw = LedoitWolf().fit(samples_np)
        return torch.tensor(lw.covariance_, dtype=samples.dtype, device=samples.device)

    def _conformal_radius(self, samples: torch.Tensor, mean: torch.Tensor,
                          cov: torch.Tensor) -> torch.Tensor:
        diff = samples - mean.unsqueeze(0)
        cov_inv = torch.linalg.inv(cov + 1e-6 * torch.eye(cov.shape[0], device=cov.device))
        mahal = torch.sqrt((diff @ cov_inv * diff).sum(dim=1).clamp(min=0))

        q_idx = int(np.ceil((1 - self.alpha) * (self.K + 1))) - 1
        q_idx = min(q_idx, self.K - 1)
        q_h = torch.sort(mahal)[0][q_idx]
        return q_h
