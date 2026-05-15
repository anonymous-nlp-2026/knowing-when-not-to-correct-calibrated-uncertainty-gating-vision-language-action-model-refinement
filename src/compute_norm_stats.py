"""Compute per-component normalization stats from CRM cache."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import json
from pathlib import Path
from cgs import ConformilizedGaussianScoring
from utils import CORRECTION_DIM, H_EFF, OBS_DIM


def _argmin_base_action_single(k_samp):
    """Select sample closest to mean for a single (K, H, D) tensor."""
    K = k_samp.shape[0]
    k_flat = k_samp.reshape(K, -1)
    mean_a = k_flat.mean(dim=0, keepdim=True)
    dists = ((k_flat - mean_a) ** 2).sum(dim=-1)
    return k_flat[dists.argmin()]


def _to_scalar_uf_single(uf):
    """Compress 40D CGS features to 3D for a single sample."""
    return torch.stack([
        uf[:30].norm(),
        uf[30:35].mean(),
        uf[35:40].mean(),
    ])


def main():
    cache_dir = Path("/root/autodl-tmp/crm_cache/")
    output_path = Path("/root/autodl-tmp/checkpoints/crm_v3/norm_stats.pt")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cgs = ConformilizedGaussianScoring(K=10, horizon=H_EFF, action_dim=CORRECTION_DIM)

    n = 0
    ba_sum = torch.zeros(H_EFF * CORRECTION_DIM)
    ba_sq = torch.zeros(H_EFF * CORRECTION_DIM)
    uf_sum = torch.zeros(CORRECTION_DIM * H_EFF + H_EFF + H_EFF)  # 40
    uf_sq = torch.zeros(CORRECTION_DIM * H_EFF + H_EFF + H_EFF)
    ufs_sum = torch.zeros(3)
    ufs_sq = torch.zeros(3)
    obs_sum = torch.zeros(OBS_DIM)
    obs_sq = torch.zeros(OBS_DIM)

    benchmarks = [d.name for d in cache_dir.iterdir()
                  if d.is_dir() and (d / "manifest.json").exists()]
    print(f"Benchmarks found: {benchmarks}")

    for bm in sorted(benchmarks):
        bm_dir = cache_dir / bm
        manifest = json.loads((bm_dir / "manifest.json").read_text())
        k_samples = torch.load(bm_dir / "k_samples.pt", map_location="cpu", weights_only=True)
        obs_features = torch.load(bm_dir / "obs_features.pt", map_location="cpu", weights_only=True)
        N = k_samples.shape[0]
        print(f"[{bm}] {N} samples  k={k_samples.shape}  obs={obs_features.shape}")

        for i in range(N):
            ba = _argmin_base_action_single(k_samples[i])
            uf, _ = cgs.compute_features(k_samples[i])
            ufs = _to_scalar_uf_single(uf)
            obs = obs_features[i]

            ba_sum += ba; ba_sq += ba ** 2
            uf_sum += uf; uf_sq += uf ** 2
            ufs_sum += ufs; ufs_sq += ufs ** 2
            obs_sum += obs; obs_sq += obs ** 2
            n += 1

            if (i + 1) % 500 == 0:
                print(f"  [{bm}] {i+1}/{N}")

    print(f"\nTotal samples: {n}")

    def _stats(s, sq, count):
        mean = s / count
        std = torch.sqrt((sq / count - mean ** 2).clamp(min=0))
        std = std.clamp(min=1e-8)
        return mean, std

    ba_mean, ba_std = _stats(ba_sum, ba_sq, n)
    uf_mean, uf_std = _stats(uf_sum, uf_sq, n)
    ufs_mean, ufs_std = _stats(ufs_sum, ufs_sq, n)
    obs_mean, obs_std = _stats(obs_sum, obs_sq, n)

    stats = {
        'ba_mean': ba_mean, 'ba_std': ba_std,
        'uf_full_mean': uf_mean, 'uf_full_std': uf_std,
        'uf_scalar_mean': ufs_mean, 'uf_scalar_std': ufs_std,
        'obs_mean': obs_mean, 'obs_std': obs_std,
        'n_samples': n,
    }
    torch.save(stats, output_path)
    print(f"\nSaved to {output_path}")

    print(f"\n=== base_action (30D) ===")
    print(f"  mean abs: {ba_mean.abs().mean():.6f}")
    print(f"  std range: [{ba_std.min():.6f}, {ba_std.max():.6f}]")

    print(f"\n=== uncertainty_features full (40D) ===")
    print(f"  per_dof_std [0:30]  mean_abs={uf_mean[:30].abs().mean():.6f}  std=[{uf_std[:30].min():.6f}, {uf_std[:30].max():.6f}]")
    print(f"  conf_radius [30:35] mean={uf_mean[30:35].mean():.6f}  std=[{uf_std[30:35].min():.6f}, {uf_std[30:35].max():.6f}]")
    print(f"  log_volume  [35:40] mean={uf_mean[35:40].mean():.6f}  std=[{uf_std[35:40].min():.6f}, {uf_std[35:40].max():.6f}]")

    print(f"\n=== uncertainty_features scalar (3D) ===")
    print(f"  mean: {ufs_mean}")
    print(f"  std:  {ufs_std}")

    print(f"\n=== obs_features (960D) ===")
    print(f"  mean abs: {obs_mean.abs().mean():.6f}")
    print(f"  std range: [{obs_std.min():.6f}, {obs_std.max():.6f}]")


if __name__ == "__main__":
    main()
