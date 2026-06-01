"""
Calibrate CRM correction magnitude as zero-overhead gating signal.

Tests whether ||CRM(obs, a) - a||₂ correlates with K=10 action variance,
enabling zero-cost uncertainty estimation from single VLA sample + CRM (0.25ms).
"""
import functools
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import sys
import json
import time
import numpy as np
import torch
from pathlib import Path

LOGFILE = "./logs/calibrate_crm_signal.log"
Path(LOGFILE).parent.mkdir(parents=True, exist_ok=True)
_logfh = open(LOGFILE, "w")

def log(msg):
    line = str(msg)
    _logfh.write(line + "\n")
    _logfh.flush()
    print(line, flush=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crm import ConformalRefinementModule
from utils import set_seed, CORRECTION_DIM, FULL_ACTION_DIM, H_EFF, OBS_DIM


def main():
    benchmark = "object"
    seed = 42
    n_rollouts = 10
    max_steps = 300
    K = 10
    crm_checkpoint = "./checkpoints/crm_best.pt"
    max_correction_norm = 0.2
    smolvla_path = "./models/smolvla"
    output_file = "logs/crm_magnitude_calibration.json"

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from smolvla_wrapper import SmolVLAWrapper
    log(f"[calibrate] Loading SmolVLA from {smolvla_path}...")
    wrapper = SmolVLAWrapper(smolvla_path, device=str(device))

    log(f"[calibrate] Loading CRM from {crm_checkpoint}...")
    crm = ConformalRefinementModule(
        action_dim=H_EFF * CORRECTION_DIM,
        uncertainty_dim=0,
        obs_dim=OBS_DIM,
        hidden_dim=256,
        max_correction_norm=max_correction_norm,
    )
    crm.load_state_dict(torch.load(crm_checkpoint, map_location=device, weights_only=False))
    crm.max_correction_norm = max_correction_norm
    crm.eval()
    crm.to(device)
    log(f"[calibrate] CRM loaded ({crm.count_parameters():,} params), max_correction_norm={max_correction_norm}")

    from evaluate import load_libero_env, get_robot_state
    log(f"[calibrate] Loading LIBERO {benchmark} environments...")
    task_envs = load_libero_env(benchmark)

    records = []
    global_step = 0
    t0 = time.time()

    for task_idx, task_info in enumerate(task_envs):
        task_name = task_info["task_description"]
        env = task_info["env"]
        init_states = task_info["init_states"]

        log(f"\n[Task {task_idx+1}/10] {task_name}")

        for r in range(n_rollouts):
            env.reset()
            obs = env.set_init_state(init_states[r % len(init_states)])
            for _ in range(5):
                obs, _, _, _ = env.step(np.zeros(7))

            ep_steps = 0
            for step in range(max_steps):
                agentview = obs["agentview_image"]
                eye_in_hand = obs.get("robot0_eye_in_hand_image", None)
                robot_state = get_robot_state(obs)

                obs_feat, samples = wrapper.get_obs_features_and_samples(
                    agentview_rgb=agentview,
                    task_name=task_name,
                    robot_state=robot_state,
                    eye_in_hand_rgb=eye_in_hand,
                    K=K,
                )
                k_full = samples[0, :, :H_EFF, :]
                k_pose = k_full[:, :, :CORRECTION_DIM]
                k_pose_flat = k_pose.reshape(K, -1)

                k10_variance = k_pose.var(dim=0).mean().item()

                obs_feat_dev = obs_feat.to(device)

                a1_flat = k_pose_flat[0].unsqueeze(0).to(device)
                with torch.no_grad():
                    a1_corrected = crm(a1_flat, None, obs_feat_dev)
                k1_correction_mag = (a1_corrected - a1_flat).norm().item()

                k2_l2_distance = (k_pose_flat[0] - k_pose_flat[1]).norm().item()

                a2_flat = k_pose_flat[1].unsqueeze(0).to(device)
                with torch.no_grad():
                    a2_corrected = crm(a2_flat, None, obs_feat_dev)
                a2_recon_err = (a2_corrected - a2_flat).norm().item()

                if k1_correction_mag <= a2_recon_err:
                    k2_correction_mag = k1_correction_mag
                else:
                    k2_correction_mag = a2_recon_err

                records.append({
                    "step": global_step,
                    "episode": task_idx * n_rollouts + r,
                    "task": task_name,
                    "k10_variance": k10_variance,
                    "k1_correction_magnitude": k1_correction_mag,
                    "k2_correction_magnitude": k2_correction_mag,
                    "k2_l2_distance": k2_l2_distance,
                })
                global_step += 1
                ep_steps += 1

                mean_action = k_full.mean(dim=0)
                action_7d = mean_action[0].cpu().numpy()
                obs, _, done, info = env.step(action_7d)

                if done or info.get("success", False):
                    break

            log(f"  T{task_idx+1} R{r+1}: {ep_steps} steps, total={global_step}, {time.time()-t0:.0f}s")

        env.close()

    elapsed = time.time() - t0
    log(f"\n[calibrate] Done: {global_step} steps collected in {elapsed:.0f}s")

    from scipy.stats import spearmanr

    k1_corr_mags = [r["k1_correction_magnitude"] for r in records]
    k2_corr_mags = [r["k2_correction_magnitude"] for r in records]
    k2_l2_dists = [r["k2_l2_distance"] for r in records]
    k10_vars = [r["k10_variance"] for r in records]

    rho1, p1 = spearmanr(k1_corr_mags, k10_vars)
    rho2, p2 = spearmanr(k2_corr_mags, k10_vars)
    rho3, p3 = spearmanr(k1_corr_mags, k2_l2_dists)
    rho4, p4 = spearmanr(k2_l2_dists, k10_vars)

    def dist_stats(arr):
        a = np.array(arr)
        return {
            "mean": float(a.mean()),
            "std": float(a.std()),
            "p25": float(np.percentile(a, 25)),
            "p50": float(np.percentile(a, 50)),
            "p75": float(np.percentile(a, 75)),
            "p90": float(np.percentile(a, 90)),
        }

    correlations = {
        "k1_corr_mag_vs_k10_var": {"rho": float(rho1), "p_value": float(p1)},
        "k2_corr_mag_vs_k10_var": {"rho": float(rho2), "p_value": float(p2)},
        "k1_corr_mag_vs_k2_l2_dist": {"rho": float(rho3), "p_value": float(p3)},
        "k2_l2_dist_vs_k10_var": {"rho": float(rho4), "p_value": float(p4)},
    }

    distributions = {
        "k1_correction_magnitude": dist_stats(k1_corr_mags),
        "k2_correction_magnitude": dist_stats(k2_corr_mags),
        "k2_l2_distance": dist_stats(k2_l2_dists),
        "k10_variance": dist_stats(k10_vars),
    }

    log(f"\n{'='*60}")
    log("Correlation Analysis")
    log(f"{'='*60}")
    for name, vals in correlations.items():
        log(f"  {name}: rho={vals['rho']:.4f}, p={vals['p_value']:.2e}")
    log(f"\n{'='*60}")
    log("Distribution Statistics")
    log(f"{'='*60}")
    for name, s in distributions.items():
        log(f"  {name}:")
        log(f"    mean={s['mean']:.6f}, std={s['std']:.6f}")
        log(f"    P25={s['p25']:.6f}, P50={s['p50']:.6f}, P75={s['p75']:.6f}, P90={s['p90']:.6f}")

    go = rho1 > 0.3 or rho2 > 0.3
    decision = "GO" if go else "NO-GO"
    log(f"\n{'='*60}")
    log(f"Decision: {decision}")
    if go:
        best_signal = "k1_correction_magnitude" if rho1 >= rho2 else "k2_correction_magnitude"
        best_rho = max(rho1, rho2)
        log(f"  Best signal: {best_signal} (rho={best_rho:.4f})")
    else:
        log(f"  Both CRM signals too weak: rho1={rho1:.4f}, rho2={rho2:.4f}")
    log(f"{'='*60}")

    result = {
        "benchmark": benchmark,
        "seed": seed,
        "n_rollouts": n_rollouts,
        "total_steps": global_step,
        "max_correction_norm": max_correction_norm,
        "correlations": correlations,
        "distributions": distributions,
        "decision": decision,
        "records": records,
    }

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    Path(output_file).write_text(json.dumps(result, indent=2))
    log(f"\n[calibrate] Results saved to {output_file}")
    _logfh.close()


if __name__ == "__main__":
    main()
