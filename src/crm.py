"""
Conformal Refinement Module (CRM).

Input: base_action (30D, argmin-selected) + uncertainty_features (40D) + obs_features (960D from VLM)
Output: refined_action (30D) = base_action + clipped_delta

Architecture: 2 hidden layers, dim 256, ReLU, residual connection.
"""
import torch
import torch.nn as nn


class ConformalRefinementModule(nn.Module):
    """Learn to refine VLA action predictions using structured uncertainty."""

    def __init__(self, action_dim: int = 30, uncertainty_dim: int = 40,
                 obs_dim: int = 960, hidden_dim: int = 256,
                 max_correction_norm: float = 1.42):
        """
        Args:
            action_dim: flattened action dim (H_eff * D = 5 * 6 = 30)
            uncertainty_dim: CGS output dim (30 + 5 + 5 = 40), 0 for obs_only
            obs_dim: VLM encoder frozen output dim (960)
            hidden_dim: hidden layer dim
        """
        super().__init__()

        self.max_correction_norm = max_correction_norm
        self.uncertainty_dim = uncertainty_dim
        input_dim = action_dim + uncertainty_dim + obs_dim

        self.register_buffer('input_mean', torch.zeros(input_dim))
        self.register_buffer('input_std', torch.ones(input_dim))

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Residual: output = base_action + delta. Init last layer near zero.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, base_action: torch.Tensor, uncertainty_features: torch.Tensor,
                obs_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            base_action: (B, 30) argmin-selected sample, flattened
            uncertainty_features: (B, 40) CGS output, or None for obs_only
            obs_features: (B, 960) VLM frozen obs features
        Returns:
            refined_action: (B, 30) corrected action
        """
        if uncertainty_features is not None:
            x = torch.cat([base_action, uncertainty_features, obs_features], dim=-1)
        else:
            x = torch.cat([base_action, obs_features], dim=-1)
        x = (x - self.input_mean) / (self.input_std + 1e-8)
        delta = self.net(x)
        # L2 norm clipping to prevent catastrophic over-correction
        delta_norm = delta.norm(dim=-1, keepdim=True)
        delta = delta * torch.clamp(self.max_correction_norm / (delta_norm + 1e-8), max=1.0)
        return base_action + delta

    def set_normalization_stats(self, mean: "torch.Tensor", std: "torch.Tensor"):
        self.input_mean.copy_(mean)
        self.input_std.copy_(std)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
