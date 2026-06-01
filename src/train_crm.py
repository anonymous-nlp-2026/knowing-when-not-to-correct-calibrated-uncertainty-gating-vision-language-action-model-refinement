"""
CRM training script.

Two-phase approach:
  Phase 1 (--precompute): Run frozen SmolVLA on LIBERO demos, cache
      (obs_features, K_samples, expert_action) to disk.
  Phase 2 (default): Load cached features, train CRM with MSE loss +
      synthetic augmentation.

Usage:
  # Phase 1: precompute features (slow, GPU-heavy)
  python train_crm.py --precompute --benchmark object

  # Phase 2: train CRM (fast)
  python train_crm.py --benchmark object --epochs 50

  # End-to-end (precompute if cache missing, then train)
  python train_crm.py --benchmark object --epochs 50 --auto
"""
import argparse
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import sys
import json
import time
import signal
import traceback
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cgs import ConformilizedGaussianScoring
from crm import ConformalRefinementModule
from utils import (
    set_seed, load_libero_demos, build_training_samples,
    synthetic_augmentation, CheckpointManager,
    CORRECTION_DIM, H_EFF, OBS_DIM, CHUNK_SIZE,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train CRM on LIBERO demos")
    p.add_argument("--benchmark", type=str, nargs='+', default=["spatial"],
                   choices=["object", "spatial", "goal", "long", "90"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str,
                   default="./checkpoints/")
    p.add_argument("--cache_dir", type=str,
                   default="./cache/")
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--obs_dim", type=int, default=OBS_DIM)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--aug_noise_min", type=float, default=0.02)
    p.add_argument("--aug_noise_max", type=float, default=0.1)
    p.add_argument("--aug_ratio", type=float, default=0.5,
                   help="fraction of batch to augment")
    p.add_argument("--precompute", action="store_true",
                   help="only precompute features, don't train")
    p.add_argument("--auto", action="store_true",
                   help="precompute if cache missing, then train")
    p.add_argument("--smolvla_path", type=str, default="./models/smolvla")
    p.add_argument("--model", type=str, default="smolvla",
                   choices=["smolvla", "pi05"],
                   help="which VLA to use for precompute")
    p.add_argument("--pi05_path", type=str,
                   default="./models/pi05")
    p.add_argument("--max_demos_per_task", type=int, default=50)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--precompute_batch_size", type=int, default=4,
                   help="batch size for SmolVLA forward during precompute")
    p.add_argument("--save_every", type=int, default=500,
                   help="save partial cache every N samples")
    p.add_argument("--shard_id", type=int, default=0,
                   help="shard ID for parallel precompute (0-based)")
    p.add_argument("--num_shards", type=int, default=1,
                   help="total number of shards for parallel precompute")
    p.add_argument("--input_mode", type=str, default="full",
                   choices=["full", "obs_only", "scalar_unc"],
                   help="full=all features, obs_only=no conformal, scalar_unc=scalar uncertainty")
    p.add_argument("--loss_type", type=str, default="mse",
                   choices=["mse", "perdof_weighted", "conformal_guided", "combined"],
                   help="loss function variant")
    p.add_argument("--lambda_conf", type=float, default=0.1,
                   help="weight for conformal-guided auxiliary loss")
    p.add_argument("--unified_obs_dim", type=int, default=None,
                   help="Zero-pad obs_features to this dim for multi-suite training")
    return p.parse_args()


# ── Precomputation ──────────────────────────────────────────────────

def precompute_features(args):
    """Run frozen SmolVLA on all demo timesteps, save features to disk."""
    benchmarks_to_run = []
    for bm in args.benchmark:
        cache_dir = Path(args.cache_dir) / bm
        if (cache_dir / "manifest.json").exists():
            print(f"[precompute] Cache exists at {cache_dir}, skipping {bm}.")
        else:
            benchmarks_to_run.append(bm)

    if not benchmarks_to_run:
        return

    if args.model == "pi05":
        from pi05_wrapper import Pi05Wrapper
        print(f"[precompute] Loading PI05 from {args.pi05_path}...")
        wrapper = Pi05Wrapper(args.pi05_path, device="cuda")
    else:
        from smolvla_wrapper import SmolVLAWrapper
        print(f"[precompute] Loading SmolVLA from {args.smolvla_path}...")
        wrapper = SmolVLAWrapper(args.smolvla_path, device="cuda")
    print(f"[precompute] VLM hidden_size = {wrapper.vlm_hidden_size}")

    for benchmark in benchmarks_to_run:
        cache_dir = Path(args.cache_dir) / benchmark
        cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"[precompute] Loading LIBERO demos ({benchmark})...")
        trajectories = load_libero_demos(benchmark,
                                         max_demos_per_task=args.max_demos_per_task)
        all_samples = build_training_samples(trajectories)
        total_all = len(all_samples)
        print(f"[precompute] {total_all} total samples from "
              f"{len(trajectories)} trajectories.")

        # Sharding: each shard takes every num_shards-th sample
        shard_indices = list(range(args.shard_id, total_all, args.num_shards))
        samples = [all_samples[i] for i in shard_indices]
        total = len(samples)
        shard_tag = f"s{args.shard_id}"

        if args.num_shards > 1:
            print(f"[precompute] Shard {args.shard_id}/{args.num_shards}: "
                  f"{total} samples")

        # Resume: find how many samples already done for this shard
        done_count = 0
        for pf in cache_dir.glob(f"partial_{shard_tag}_*.pt"):
            parts = pf.stem.split("_")
            end_idx = int(parts[-1])
            done_count = max(done_count, end_idx)

        if done_count >= total:
            print(f"[precompute] Shard {args.shard_id} already complete "
                  f"({done_count} samples)")
            _try_merge(cache_dir, total_all, args)
            continue

        if done_count > 0:
            print(f"[precompute] Resuming from sample {done_count}/{total}")

        bs = args.precompute_batch_size
        save_every = args.save_every
        obs_buf, k_buf, exp_buf = [], [], []
        buf_start = done_count
        t0 = time.time()

        for i in range(done_count, total, bs):
            batch_end = min(i + bs, total)
            batch_samples = samples[i:batch_end]

            if args.model == "pi05":
                obs_feats, k_samples_list = [], []
                for s in batch_samples:
                    feat, samp = wrapper.get_obs_features_and_samples(
                        s["agentview_rgb"], s["task_name"], s["ee_states"],
                        eye_in_hand_rgb=s.get("eye_in_hand_rgb"), K=args.K
                    )
                    obs_feats.append(feat)
                    k_samples_list.append(samp)
                obs_feat = torch.cat(obs_feats, dim=0)
                k_samples = torch.cat(k_samples_list, dim=0)
            else:
                obs_feat, k_samples = wrapper.get_obs_features_and_samples_batch(
                    batch_samples, K=args.K
                )

            # Slice to pose-only for CRM training
            k_trunc = k_samples[:, :, :H_EFF, :CORRECTION_DIM]
            obs_buf.append(obs_feat.cpu())
            k_buf.append(k_trunc.cpu())
            exp_buf.append(torch.stack([
                torch.tensor(s["expert_action"], dtype=torch.float32)
                for s in batch_samples
            ]))

            processed = batch_end
            buf_count = sum(t.shape[0] for t in obs_buf)

            if processed % 100 < bs or processed == total:
                elapsed = time.time() - t0
                n_new = processed - done_count
                rate = elapsed / max(n_new, 1)
                remaining = total - processed
                eta = rate * remaining
                print(f"[precompute] [{benchmark}] {processed}/{total} "
                      f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining, "
                      f"{rate:.2f}s/sample)")

            if buf_count >= save_every or processed == total:
                buf_end = buf_start + buf_count
                partial_path = cache_dir / f"partial_{shard_tag}_{buf_start}_{buf_end}.pt"
                torch.save({
                    "obs_features": torch.cat(obs_buf),
                    "k_samples": torch.cat(k_buf),
                    "expert_actions": torch.cat(exp_buf),
                }, partial_path)
                print(f"[precompute] Saved [{buf_start}:{buf_end}] -> "
                      f"{partial_path.name}")
                obs_buf, k_buf, exp_buf = [], [], []
                buf_start = buf_end

        _try_merge(cache_dir, total_all, args)


def _try_merge(cache_dir, total_samples, args):
    """Merge partial files from all shards into final cache if complete."""
    for sid in range(args.num_shards):
        stag = f"s{sid}"
        expected = len(range(sid, total_samples, args.num_shards))
        done = 0
        for pf in cache_dir.glob(f"partial_{stag}_*.pt"):
            end_idx = int(pf.stem.split("_")[-1])
            done = max(done, end_idx)
        if done < expected:
            print(f"[merge] Shard {sid} not done ({done}/{expected}), "
                  f"skipping merge.")
            return

    print(f"[merge] All {args.num_shards} shard(s) complete. Merging...")

    shard_data = {}
    for sid in range(args.num_shards):
        stag = f"s{sid}"
        partials = sorted(
            cache_dir.glob(f"partial_{stag}_*.pt"),
            key=lambda p: int(p.stem.split("_")[-2])
        )
        obs_parts, k_parts, exp_parts = [], [], []
        for pf in partials:
            data = torch.load(pf, weights_only=True)
            obs_parts.append(data["obs_features"])
            k_parts.append(data["k_samples"])
            exp_parts.append(data["expert_actions"])
        shard_data[sid] = (
            torch.cat(obs_parts),
            torch.cat(k_parts),
            torch.cat(exp_parts),
        )

    if args.num_shards == 1:
        obs_merged, k_merged, exp_merged = shard_data[0]
    else:
        n = total_samples
        obs_merged = torch.zeros(n, shard_data[0][0].shape[1])
        k_merged = torch.zeros(n, *shard_data[0][1].shape[1:])
        exp_merged = torch.zeros(n, *shard_data[0][2].shape[1:])
        for sid in range(args.num_shards):
            indices = list(range(sid, n, args.num_shards))
            obs_merged[indices] = shard_data[sid][0]
            k_merged[indices] = shard_data[sid][1]
            exp_merged[indices] = shard_data[sid][2]

    torch.save(obs_merged, cache_dir / "obs_features.pt")
    torch.save(k_merged, cache_dir / "k_samples.pt")
    torch.save(exp_merged, cache_dir / "expert_actions.pt")

    manifest = {
        "benchmark": cache_dir.name,
        "n_samples": int(obs_merged.shape[0]),
        "K": args.K,
        "obs_dim": int(obs_merged.shape[1]),
        "action_dim": CORRECTION_DIM,
        "h_eff": H_EFF,
    }
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[merge] Saved {obs_merged.shape[0]} samples + manifest to {cache_dir}")


# ── Dataset ─────────────────────────────────────────────────────────

class CRMDataset(Dataset):
    """Dataset of precomputed (obs_features, k_samples, expert_action) tuples."""

    def __init__(self, obs_features, k_samples, expert_actions):
        self.obs_features = obs_features    # (N, 960)
        self.k_samples = k_samples          # (N, K, H_eff, CORRECTION_DIM)
        self.expert_actions = expert_actions  # (N, H_eff, CORRECTION_DIM)

    def __len__(self):
        return self.obs_features.shape[0]

    def __getitem__(self, idx):
        return {
            "obs_features": self.obs_features[idx],
            "k_samples": self.k_samples[idx],
            "expert_action": self.expert_actions[idx],
        }


# ── Training ────────────────────────────────────────────────────────

def _argmin_base_action(k_samp: "torch.Tensor") -> "torch.Tensor":
    """Select the sample closest to the mean (argmin Euclidean) per batch element."""
    B, K = k_samp.shape[0], k_samp.shape[1]
    k_flat = k_samp.reshape(B, K, -1)        # (B, K, 30)
    mean_a = k_flat.mean(dim=1, keepdim=True)  # (B, 1, 30)
    dists = ((k_flat - mean_a) ** 2).sum(dim=-1)  # (B, K)
    best_idx = dists.argmin(dim=1)             # (B,)
    return k_flat[torch.arange(B, device=k_flat.device), best_idx]  # (B, 30)


def _argmin_to_expert(k_samp, expert_flat):
    """Select the K-sample closest to expert action per batch element."""
    B, K = k_samp.shape[0], k_samp.shape[1]
    k_flat = k_samp.reshape(B, K, -1)
    dists = ((k_flat - expert_flat.unsqueeze(1)) ** 2).sum(dim=-1)
    best_idx = dists.argmin(dim=1)
    return k_flat[torch.arange(B, device=k_flat.device), best_idx]


def _compute_loss(refined, expert_flat, ba, k_samp, loss_type, dof_weights, lambda_conf):
    """Compute CRM loss based on loss_type."""
    if loss_type == "mse":
        return F.mse_loss(refined, expert_flat)
    elif loss_type == "perdof_weighted":
        return (dof_weights * (refined - expert_flat) ** 2).mean()
    else:
        # conformal_guided or combined
        argmin_expert = _argmin_to_expert(k_samp, expert_flat)
        nonconformity = torch.norm(argmin_expert - expert_flat, dim=-1)
        delta_a = refined - ba
        correction_mag = torch.norm(delta_a, dim=-1)
        nc_norm = nonconformity / (nonconformity.mean() + 1e-8)
        cm_norm = correction_mag / (correction_mag.mean() + 1e-8)
        loss_conf = F.mse_loss(cm_norm, nc_norm.detach())
        if loss_type == "conformal_guided":
            return F.mse_loss(refined, expert_flat) + lambda_conf * loss_conf
        else:  # combined
            loss_pdw = (dof_weights * (refined - expert_flat) ** 2).mean()
            return loss_pdw + lambda_conf * loss_conf


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_obs_features = []
    all_k_samples = []
    all_expert_actions = []
    manifest = None

    for bm in args.benchmark:
        cache_dir = Path(args.cache_dir) / bm
        manifest_path = cache_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"No precomputed cache at {cache_dir}. "
                f"Run with --precompute first, or use --auto."
            )
        manifest = json.loads(manifest_path.read_text())
        print(f"[train] Loading cache ({bm}): {manifest['n_samples']} samples, "
              f"K={manifest['K']}, obs_dim={manifest['obs_dim']}")
        all_obs_features.append(torch.load(cache_dir / "obs_features.pt", weights_only=True))
        all_k_samples.append(torch.load(cache_dir / "k_samples.pt", weights_only=True))
        all_expert_actions.append(torch.load(cache_dir / "expert_actions.pt", weights_only=True))

    obs_features = torch.cat(all_obs_features)
    k_samples = torch.cat(all_k_samples)
    expert_actions = torch.cat(all_expert_actions)

    # Zero-pad obs_features for multi-suite unified training
    if args.unified_obs_dim and obs_features.shape[1] < args.unified_obs_dim:
        pad_size = args.unified_obs_dim - obs_features.shape[1]
        obs_features = F.pad(obs_features, (0, pad_size))
        print(f"[train] Zero-padded obs_features to {obs_features.shape[1]} "
              f"(+{pad_size} dims)")

    print(f"[train] Total: {obs_features.shape[0]} samples from "
          f"{len(args.benchmark)} benchmark(s)")

    # Train/val split
    N = obs_features.shape[0]
    n_val = int(N * args.val_split)
    n_train = N - n_val
    perm = torch.randperm(N)
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    # Pre-compute DOF weights for perdof_weighted/combined
    dof_weights = None
    if args.loss_type in ("perdof_weighted", "combined"):
        expert_flat_train = expert_actions[train_idx].reshape(-1, H_EFF * CORRECTION_DIM)
        dof_variance = expert_flat_train.var(dim=0)
        dof_weights = 1.0 / (dof_variance + 1e-8)
        dof_weights = dof_weights / dof_weights.sum() * (H_EFF * CORRECTION_DIM)
        dof_weights = dof_weights.to(device)
        print(f"[train] DOF weights: min={dof_weights.min():.4f}, max={dof_weights.max():.4f}")

    train_ds = CRMDataset(obs_features[train_idx], k_samples[train_idx],
                          expert_actions[train_idx])
    val_ds = CRMDataset(obs_features[val_idx], k_samples[val_idx],
                        expert_actions[val_idx])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2,
                            shuffle=False, num_workers=2, pin_memory=True)

    # CGS + CRM — operate on 6D pose only (CORRECTION_DIM)
    cgs = ConformilizedGaussianScoring(
        K=manifest["K"], horizon=H_EFF, action_dim=CORRECTION_DIM
    )
    # uncertainty_dim depends on input_mode
    unc_dim_map = {"full": CORRECTION_DIM * H_EFF + H_EFF + H_EFF,  # 40
                   "obs_only": 0,
                   "scalar_unc": 3}
    unc_dim = unc_dim_map[args.input_mode]
    obs_only = (args.input_mode == "obs_only")
    scalar_unc = (args.input_mode == "scalar_unc")

    crm = ConformalRefinementModule(
        action_dim=H_EFF * CORRECTION_DIM,  # 30
        uncertainty_dim=unc_dim,
        obs_dim=manifest["obs_dim"],
        hidden_dim=args.hidden_dim,
    ).to(device)

    print(f"[train] CRM parameters: {crm.count_parameters():,}")

    optimizer = torch.optim.Adam(crm.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    loss_fn = nn.MSELoss()
    ckpt_mgr = CheckpointManager(args.output_dir, max_keep=5)

    noise_scale = torch.linspace(
        args.aug_noise_min, args.aug_noise_max, CORRECTION_DIM
    ).to(device)

    def _to_scalar_uf(uf):
        """Compress 40D CGS features to 3D: (norm_std, mean_radius, mean_logvol)."""
        return torch.stack([
            uf[:, :30].norm(dim=1),
            uf[:, 30:35].mean(dim=1),
            uf[:, 35:40].mean(dim=1),
        ], dim=1)


    # ── Compute input normalization stats (training set only) ──
    print("[train] Computing input normalization stats...")
    _stat_loader = DataLoader(train_ds, batch_size=args.batch_size * 4,
                              shuffle=False, num_workers=2)
    _input_dim = crm.input_mean.shape[0]
    _sum_x = torch.zeros(_input_dim)
    _sum_x2 = torch.zeros(_input_dim)
    _n_total = 0
    with torch.no_grad():
        for _batch in _stat_loader:
            _uf, _ = cgs.compute_features_batch(_batch["k_samples"])
            if scalar_unc:
                _uf = _to_scalar_uf(_uf)
            _ba = _argmin_base_action(_batch["k_samples"])
            if obs_only:
                _x = torch.cat([_ba, _batch["obs_features"]], dim=-1)
            else:
                _x = torch.cat([_ba, _uf, _batch["obs_features"]], dim=-1)
            _x = torch.nan_to_num(_x, nan=0.0, posinf=1e6, neginf=-1e6)
            _sum_x += _x.sum(dim=0)
            _sum_x2 += (_x ** 2).sum(dim=0)
            _n_total += _x.shape[0]
    _input_mean = _sum_x / _n_total
    _input_std = ((_sum_x2 / _n_total) - _input_mean ** 2).clamp(min=0).sqrt()
    _input_mean = torch.nan_to_num(_input_mean, nan=0.0, posinf=0.0, neginf=0.0)
    _input_std = torch.nan_to_num(_input_std, nan=1.0, posinf=1.0, neginf=1.0)
    _input_std = _input_std.clamp(min=1e-6)
    crm.set_normalization_stats(_input_mean, _input_std)
    print(f"[train] Norm stats set: mean [{_input_mean.min():.4f}, {_input_mean.max():.4f}], "
          f"std [{_input_std.min():.4f}, {_input_std.max():.4f}]")
    del _stat_loader, _sum_x, _sum_x2, _input_mean, _input_std

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # ── Train ──
        crm.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            obs_feat = batch["obs_features"].to(device)       # (B, 960)
            k_samp = batch["k_samples"].to(device)             # (B, K, H, D)
            expert = batch["expert_action"].to(device)         # (B, H, D)
            expert_flat = expert.reshape(expert.shape[0], -1)  # (B, 30)

            # CGS (uncertainty features only — base action is argmin-selected, not mean)
            uf, _mean = cgs.compute_features_batch(k_samp)
            if scalar_unc:
                uf = _to_scalar_uf(uf)
            uf = torch.nan_to_num(uf, nan=0.0, posinf=1e6, neginf=-1e6).to(device)
            ba = _argmin_base_action(k_samp).to(device)  # (B, 30)

            # CRM forward
            refined = crm(ba, None if obs_only else uf, obs_feat)
            loss = _compute_loss(refined, expert_flat, ba, k_samp,
                                 args.loss_type, dof_weights, args.lambda_conf)

            # Synthetic augmentation: perturb inputs, keep clean target
            n_aug = int(obs_feat.shape[0] * args.aug_ratio)
            if n_aug > 0:
                ba_noise = noise_scale.repeat(H_EFF)
                aug_ba = ba[:n_aug] + torch.randn_like(ba[:n_aug]) * ba_noise
                aug_uf = uf[:n_aug] + torch.randn_like(uf[:n_aug]) * noise_scale.mean() * 0.1
                aug_refined = crm(aug_ba, None if obs_only else aug_uf, obs_feat[:n_aug])
                aug_loss = _compute_loss(aug_refined, expert_flat[:n_aug], aug_ba,
                                         k_samp[:n_aug], args.loss_type, dof_weights,
                                         args.lambda_conf)
                loss = loss + 0.5 * aug_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(crm.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            if (batch_idx + 1) % args.log_every == 0:
                print(f"  epoch {epoch} batch {batch_idx+1}/{len(train_loader)} "
                      f"loss={loss.item():.6f}")

        scheduler.step()
        avg_train_loss = epoch_loss / max(n_batches, 1)

        # ── Validate ──
        crm.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                obs_feat = batch["obs_features"].to(device)
                k_samp = batch["k_samples"].to(device)
                expert = batch["expert_action"].to(device)
                expert_flat = expert.reshape(expert.shape[0], -1)

                uf, _mean = cgs.compute_features_batch(k_samp)
                if scalar_unc:
                    uf = _to_scalar_uf(uf)
                uf = uf.to(device)
                ba = _argmin_base_action(k_samp).to(device)

                refined = crm(ba, None if obs_only else uf, obs_feat)
                val_loss += _compute_loss(refined, expert_flat, ba, k_samp,
                                          args.loss_type, dof_weights,
                                          args.lambda_conf).item()
                n_val_batches += 1

        avg_val_loss = val_loss / max(n_val_batches, 1)

        print(f"Epoch {epoch}/{args.epochs}  "
              f"train_loss={avg_train_loss:.6f}  "
              f"val_loss={avg_val_loss:.6f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        metrics = {"train_loss": avg_train_loss, "val_loss": avg_val_loss}
        ckpt_mgr.save(crm, optimizer, epoch, metrics)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(crm.state_dict(),
                       Path(args.output_dir) / "crm_best.pt")
            print(f"  → New best val_loss: {best_val_loss:.6f}")

    print(f"\n[train] Done. Best val_loss={best_val_loss:.6f}")
    print(f"[train] Checkpoints at {args.output_dir}")


def main():
    args = parse_args()
    print(f"[train_crm] args: {vars(args)}")
    print(f"[train_crm] device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    if args.precompute:
        precompute_features(args)
        return

    if args.auto:
        needs_precompute = any(
            not (Path(args.cache_dir) / bm / "manifest.json").exists()
            for bm in args.benchmark
        )
        if needs_precompute:
            print("[auto] Cache not found for some benchmarks, precomputing...")
            precompute_features(args)

    train(args)


if __name__ == "__main__":
    main()
