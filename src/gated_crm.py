"""
Gated Conformal Refinement Module (GatedCRM).

Extends CRM with per-DOF uncertainty gating: correction is zeroed out on
dimensions where per-DOF std (from CGS) is below a threshold, preventing
over-correction on confident dimensions.

Architecture: identical MLP to CRM (2 hidden layers, dim 256, ReLU).
Compatible with existing CRM checkpoints — gate parameters are new, MLP
weights load directly via state_dict.
"""
import torch
import torch.nn as nn


class GatedCRM(nn.Module):
    """CRM with per-DOF uncertainty gating.

    When per_dof_std[d] < gate_threshold for dimension d,
    correction[d] → 0 (pass-through base action).
    Only applies correction on high-uncertainty dimensions.
    """

    def __init__(self, action_dim: int = 30, uncertainty_dim: int = 40,
                 obs_dim: int = 960, hidden_dim: int = 256,
                 max_correction_norm: float = 1.42,
                 gate_threshold: float = 0.1,
                 gate_mode: str = 'hard',
                 gate_temperature: float = 0.05):
        """
        Args:
            action_dim: flattened action dim (H_eff * D = 5 * 6 = 30)
            uncertainty_dim: CGS output dim (30 + 5 + 5 = 40)
            obs_dim: VLM encoder frozen output dim (960)
            hidden_dim: hidden layer dim
            max_correction_norm: L2 clipping bound for delta
            gate_threshold: per-DOF std below which correction is zeroed
            gate_mode: 'hard' (binary 0/1) or 'soft' (sigmoid scaling)
            gate_temperature: controls sigmoid sharpness in soft mode
        """
        super().__init__()

        self.action_dim = action_dim
        self.max_correction_norm = max_correction_norm
        self.uncertainty_dim = uncertainty_dim
        self.gate_threshold = gate_threshold
        self.gate_mode = gate_mode
        self.gate_temperature = gate_temperature

        input_dim = action_dim + uncertainty_dim + obs_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, base_action: torch.Tensor, uncertainty_features: torch.Tensor,
                obs_features: torch.Tensor,
                per_dof_uncertainty: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            base_action: (B, 30) argmin-selected sample, flattened
            uncertainty_features: (B, 40) CGS output
            obs_features: (B, 960) VLM frozen obs features
            per_dof_uncertainty: (B, 30) per-DOF std from CGS for gating.
                If None, extracts from uncertainty_features[:, :action_dim].
        Returns:
            refined_action: (B, 30)
        """
        if uncertainty_features is not None:
            x = torch.cat([base_action, uncertainty_features, obs_features], dim=-1)
        else:
            x = torch.cat([base_action, obs_features], dim=-1)

        delta = self.net(x)

        # L2 norm clipping
        delta_norm = delta.norm(dim=-1, keepdim=True)
        delta = delta * torch.clamp(self.max_correction_norm / (delta_norm + 1e-8), max=1.0)

        # Per-DOF gating
        if per_dof_uncertainty is None and uncertainty_features is not None:
            per_dof_uncertainty = uncertainty_features[:, :self.action_dim]

        if per_dof_uncertainty is not None:
            if self.gate_mode == 'hard':
                gate = (per_dof_uncertainty > self.gate_threshold).float()
            elif self.gate_mode == 'soft':
                gate = torch.sigmoid(
                    (per_dof_uncertainty - self.gate_threshold) / self.gate_temperature
                )
            else:
                raise ValueError(f"Unknown gate_mode: {self.gate_mode}")
            delta = delta * gate

        return base_action + delta

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
